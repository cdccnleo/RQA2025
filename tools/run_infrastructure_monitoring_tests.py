#!/usr/bin/env python3
"""
基础设施层监控测试运行脚本
解决线程超时和清理问题
"""

import os
import sys
import subprocess
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """设置测试环境"""
    # 设置环境变量
    os.environ['PYTHONPATH'] = str(project_root)
    os.environ['PYTEST_TIMEOUT'] = '30'
    os.environ['PYTEST_ADDOPTS'] = '--timeout=30 --timeout-method=thread'

    # 设置线程超时
    import threading
    threading.TIMEOUT_MAX = 30

    print("测试环境设置完成")
    print(f"项目根目录: {project_root}")
    print(f"Python路径: {os.environ['PYTHONPATH']}")


def run_tests_with_timeout():
    """运行测试并处理超时"""
    test_dir = project_root / "tests" / "unit" / "infrastructure" / "monitoring"

    if not test_dir.exists():
        print(f"测试目录不存在: {test_dir}")
        return False

    # 查找测试文件
    test_files = list(test_dir.glob("test_*_fixed.py"))
    if not test_files:
        print("未找到修复版本的测试文件")
        return False

    print(f"找到测试文件: {[f.name for f in test_files]}")

    # 运行测试
    success = True
    for test_file in test_files:
        print(f"\n正在运行测试: {test_file.name}")

        try:
            # 使用subprocess运行测试，设置超时
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--timeout=30",
                "--timeout-method=thread",
                "--disable-warnings"
            ],
                cwd=str(project_root),
                timeout=60,  # 整体超时60秒
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✅ {test_file.name} 测试通过")
                print("输出:")
                print(result.stdout)
            else:
                print(f"❌ {test_file.name} 测试失败")
                print("错误输出:")
                print(result.stderr)
                success = False

        except subprocess.TimeoutExpired:
            print(f"⏰ {test_file.name} 测试超时")
            success = False
        except Exception as e:
            print(f"💥 {test_file.name} 测试异常: {e}")
            success = False

    return success


def run_specific_test(test_name):
    """运行特定的测试"""
    test_dir = project_root / "tests" / "unit" / "infrastructure" / "monitoring"
    test_file = test_dir / f"test_{test_name}.py"

    if not test_file.exists():
        print(f"测试文件不存在: {test_file}")
        return False

    print(f"运行特定测试: {test_file.name}")

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short",
            "--timeout=30",
            "--timeout-method=thread"
        ],
            cwd=str(project_root),
            timeout=60,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ 测试通过")
            print("输出:")
            print(result.stdout)
            return True
        else:
            print("❌ 测试失败")
            print("错误输出:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 测试超时")
        return False
    except Exception as e:
        print(f"💥 测试异常: {e}")
        return False


def cleanup_threads():
    """清理残留线程"""
    import threading

    print("清理残留线程...")
    main_thread = threading.main_thread()

    for thread in threading.enumerate():
        if thread is not main_thread and thread.is_alive():
            try:
                print(f"等待线程结束: {thread.name}")
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print(f"线程仍在运行: {thread.name}")
            except Exception as e:
                print(f"清理线程异常: {e}")


def main():
    """主函数"""
    print("=" * 60)
    print("基础设施层监控测试运行脚本")
    print("解决线程超时和清理问题")
    print("=" * 60)

    # 设置环境
    setup_environment()

    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "cleanup":
            cleanup_threads()
            return
        elif sys.argv[1] == "specific":
            if len(sys.argv) > 2:
                test_name = sys.argv[2]
                success = run_specific_test(test_name)
            else:
                print("请指定测试名称")
                return
        else:
            print(f"未知参数: {sys.argv[1]}")
            print(
                "用法: python run_infrastructure_monitoring_tests.py [cleanup|specific <test_name>]")
            return
    else:
        # 运行所有修复版本的测试
        success = run_tests_with_timeout()

    # 清理
    cleanup_threads()

    # 输出结果
    print("\n" + "=" * 60)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("💥 部分测试失败或超时")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n用户中断，正在清理...")
        cleanup_threads()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n脚本异常: {e}")
        cleanup_threads()
        sys.exit(1)

#!/usr/bin/env python3
"""
RQA2025快速启动脚本
用于开发环境快速启动和验证系统

使用方法:
python scripts/quick_start.py

或者直接运行:
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_environment():
    """检查环境"""
    print("🔍 检查环境...")

    # 检查Python版本
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"❌ Python版本过低: {version.major}.{version.minor}.{version.micro} (需要3.9+)")
        return False

    # 检查关键依赖
    required_packages = ['fastapi', 'uvicorn', 'numpy', 'pandas', 'pytest']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")

    if missing_packages:
        print(f"\\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False

    return True

def start_server():
    """启动服务器"""
    print("\\n🚀 启动RQA2025服务器...")

    try:
        # 使用uvicorn启动服务器
        cmd = [
            sys.executable, "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ]

        print(f"执行命令: {' '.join(cmd)}")
        print("服务器将在 http://localhost:8000 启动")
        print("按 Ctrl+C 停止服务器\\n")

        # 启动服务器
        process = subprocess.Popen(cmd, cwd=os.getcwd())

        # 等待几秒让服务器启动
        time.sleep(3)

        # 检查服务器是否成功启动
        if process.poll() is None:
            print("✅ 服务器启动成功!")
            print("📖 API文档: http://localhost:8000/docs")
            print("🔍 健康检查: http://localhost:8000/health")

            # 自动打开浏览器
            try:
                webbrowser.open("http://localhost:8000/docs")
            except:
                pass  # 忽略浏览器打开失败

            # 等待用户输入
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\\n\\n🛑 正在停止服务器...")
                process.terminate()
                process.wait()
                print("✅ 服务器已停止")

        else:
            print("❌ 服务器启动失败")
            return False

    except FileNotFoundError:
        print("❌ 未找到uvicorn，请确保已安装: pip install uvicorn")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        return False

    return True

def run_quick_test():
    """运行快速测试"""
    print("\\n🧪 运行快速测试...")

    try:
        # 运行我们新创建的测试
        test_files = [
            "tests/unit/optimization/test_system_optimization_coverage_boost.py::TestCPUOptimization::test_cpu_usage_monitoring",
            "tests/unit/automation/test_automation_layer_coverage_boost.py::TestAutomatedTrading::test_market_making_automation",
            "tests/unit/testing/test_testing_layer_quality_boost.py::TestTestingInfrastructure::test_test_framework_core",
            "tests/unit/utils/test_utils_layer_validation_boost.py::TestBacktestUtils::test_backtest_performance_calculator"
        ]

        for test_file in test_files:
            print(f"运行测试: {test_file}")
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_file,
                "-v", "--tb=short", "--disable-warnings"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print("✅ 测试通过")
            else:
                print("❌ 测试失败")
                print(result.stdout)
                print(result.stderr)
                return False

        print("✅ 所有快速测试通过!")
        return True

    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False

def show_menu():
    """显示菜单"""
    print("\\n🎯 RQA2025快速启动工具")
    print("=" * 30)
    print("1. 环境检查")
    print("2. 运行测试")
    print("3. 启动服务器")
    print("4. 完整流程 (检查 → 测试 → 启动)")
    print("5. 部署验证")
    print("0. 退出")
    print()

    while True:
        try:
            choice = input("请选择操作 (0-5): ").strip()

            if choice == "1":
                if check_environment():
                    print("✅ 环境检查通过!")
                else:
                    print("❌ 环境检查失败!")
            elif choice == "2":
                if run_quick_test():
                    print("✅ 测试通过!")
                else:
                    print("❌ 测试失败!")
            elif choice == "3":
                start_server()
            elif choice == "4":
                print("执行完整流程...")
                if not check_environment():
                    print("❌ 环境检查失败，停止流程")
                    continue

                if not run_quick_test():
                    print("❌ 测试失败，停止流程")
                    continue

                start_server()
            elif choice == "5":
                print("运行部署验证...")
                result = subprocess.run([
                    sys.executable, "scripts/production_deployment_verification.py"
                ])
                if result.returncode == 0:
                    print("✅ 部署验证通过!")
                else:
                    print("❌ 部署验证失败!")
            elif choice == "0":
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")

        except KeyboardInterrupt:
            print("\\n\\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 操作失败: {str(e)}")

        print()

def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 命令行模式
        command = sys.argv[1]

        if command == "check":
            success = check_environment()
        elif command == "test":
            success = run_quick_test()
        elif command == "start":
            success = start_server()
        elif command == "verify":
            result = subprocess.run([
                sys.executable, "scripts/production_deployment_verification.py"
            ])
            success = result.returncode == 0
        else:
            print("使用方法:")
            print("  python scripts/quick_start.py check   # 环境检查")
            print("  python scripts/quick_start.py test    # 运行测试")
            print("  python scripts/quick_start.py start   # 启动服务器")
            print("  python scripts/quick_start.py verify  # 部署验证")
            print("  python scripts/quick_start.py         # 交互模式")
            return

        sys.exit(0 if success else 1)
    else:
        # 交互模式
        show_menu()

if __name__ == "__main__":
    main()

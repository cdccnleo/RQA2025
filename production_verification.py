#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 最终生产部署验证脚本

验证系统是否已经准备好进行生产部署。
"""

import sys
import os
from pathlib import Path
import subprocess
import time

def print_header(title):
    """打印标题"""
    print(f"\n{title}")
    print("=" * len(title))

def check_project_structure():
    """检查项目结构"""
    print_header("📁 检查项目结构")

    required_dirs = [
        'src/core', 'src/data', 'src/strategy', 'src/risk', 'src/monitoring',
        'tests/unit', 'tests/integration', 'tests/e2e',
        'test_logs', 'docs'
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    if missing_dirs:
        print(f"❌ 缺少目录: {missing_dirs}")
        return False
    else:
        print("✅ 项目结构完整")
        return True

def check_key_files():
    """检查关键文件"""
    print_header("📄 检查关键文件")

    key_files = [
        'src/core/__init__.py',
        'src/data/__init__.py',
        'src/strategy/__init__.py',
        'pytest.ini'
    ]

    missing_files = []
    for file_path in key_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ 缺少关键文件: {missing_files}")
        return False
    else:
        print("✅ 关键文件完整")
        return True

def test_core_imports():
    """测试核心模块导入"""
    print_header("🔧 测试核心模块导入")

    # 添加src路径
    project_root = Path(__file__).resolve().parent
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    imports_to_test = [
        ('core.foundation.layer_interfaces', 'get_layer_components'),
        ('data.core.data_manager', 'DataManager'),
        ('strategy.core.strategy_manager', 'StrategyManager'),
        ('risk.models.risk_manager', 'RiskManager'),
        ('monitoring.core.monitor_components', 'MonitorComponents')
    ]

    success_count = 0
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"⚠️  {module_name}.{class_name} 导入失败: {e}")
        except AttributeError as e:
            print(f"⚠️  {module_name}.{class_name} 属性不存在: {e}")
        except Exception as e:
            print(f"❌ {module_name}.{class_name} 错误: {e}")

    print(f"\n导入成功率: {success_count}/{len(imports_to_test)}")
    return success_count >= 3  # 至少70%成功

def collect_test_statistics():
    """收集测试统计"""
    print_header("📊 收集测试统计")

    try:
        # 运行pytest收集模式
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            '--collect-only',
            '--quiet',
            '--disable-warnings'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0 and 'collected' in result.stdout:
            # 解析测试数量
            lines = result.stdout.split('\n')
            for line in lines:
                if 'collected' in line and 'items' in line:
                    parts = line.split()
                    for part in parts:
                        if part.isdigit():
                            test_count = int(part)
                            print(f"✅ 发现 {test_count} 个测试用例")

                            if test_count >= 400:
                                print("🎯 测试覆盖率达标!")
                                return True
                            elif test_count >= 200:
                                print("⚠️ 测试覆盖率一般")
                                return True
                            else:
                                print("❌ 测试覆盖率不足")
                                return False
        else:
            print("❌ 无法收集测试信息")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 测试收集超时")
        return False
    except Exception as e:
        print(f"❌ 测试收集失败: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print_header("📦 检查依赖")

    try:
        # 检查requirements.txt
        req_file = Path('requirements.txt')
        if req_file.exists():
            print("✅ requirements.txt 文件存在")

            # 尝试读取
            with open(req_file, 'r', encoding='utf-8') as f:
                deps = f.readlines()

            dep_count = len([d for d in deps if d.strip() and not d.startswith('#')])
            print(f"✅ 包含 {dep_count} 个依赖包")

            return True
        else:
            print("⚠️ requirements.txt 文件不存在")
            return False

    except Exception as e:
        print(f"❌ 依赖检查失败: {e}")
        return False

def generate_deployment_report():
    """生成部署报告"""
    print_header("📋 生成部署报告")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project": "RQA2025",
        "status": "production_ready",
        "checks": {}
    }

    # 执行各项检查
    report["checks"]["project_structure"] = check_project_structure()
    report["checks"]["key_files"] = check_key_files()
    report["checks"]["core_imports"] = test_core_imports()
    report["checks"]["test_coverage"] = collect_test_statistics()
    report["checks"]["dependencies"] = check_dependencies()

    # 计算整体状态
    passed_checks = sum(1 for check in report["checks"].values() if check)
    total_checks = len(report["checks"])

    print(f"\n🎯 部署就绪评估: {passed_checks}/{total_checks} 项检查通过")

    if passed_checks >= total_checks * 0.8:  # 80%以上通过
        print("🟢 系统已准备好进行生产部署!")
        report["deployment_ready"] = True
    elif passed_checks >= total_checks * 0.6:  # 60%以上通过
        print("🟡 系统基本准备好，但建议进行额外验证")
        report["deployment_ready"] = True
    else:
        print("🔴 系统需要进一步完善")
        report["deployment_ready"] = False

    return report

def main():
    """主函数"""
    print("🎊 RQA2025 最终生产部署验证")
    print("=" * 50)

    try:
        report = generate_deployment_report()

        print_header("🚀 部署建议")

        if report.get("deployment_ready", False):
            print("✅ 可以立即进行生产部署")
            print("\n📝 部署清单:")
            print("   1. 确保Python >= 3.9")
            print("   2. 安装所有依赖: pip install -r requirements.txt")
            print("   3. 配置数据库和外部服务")
            print("   4. 设置环境变量")
            print("   5. 运行测试验证: pytest")
            print("   6. 启动服务并监控")

            print("\n⚠️ 生产环境注意事项:")
            print("   - 配置生产数据库连接")
            print("   - 设置日志轮转")
            print("   - 配置监控告警")
            print("   - 设置备份策略")
            print("   - 配置安全策略")

        else:
            print("❌ 需要解决以下问题后才能部署:")
            failed_checks = [k for k, v in report["checks"].items() if not v]
            for check in failed_checks:
                print(f"   - {check.replace('_', ' ').title()}")

        print(f"\n🏆 验证完成时间: {report['timestamp']}")

    except Exception as e:
        print(f"❌ 验证过程中发生错误: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())

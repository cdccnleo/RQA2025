#!/usr/bin/env python3
"""
基础设施层测试验证脚本

验证所有基础设施层测试的修复效果
"""

import sys
import subprocess
import time
from pathlib import Path


class InfrastructureTestValidator:
    """基础设施测试验证器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results = {}

    def run_validation(self):
        """运行完整验证"""
        print("=== 基础设施层测试验证器 ===\n")

        validations = [
            ("缓存系统测试", self.validate_cache_tests),
            ("并发测试验证", self.validate_concurrent_tests),
            ("接口定义检查", self.validate_interface_definitions),
            ("超时机制验证", self.validate_timeout_mechanisms),
            ("整体测试运行", self.validate_full_test_suite)
        ]

        all_passed = True
        for validation_name, validation_func in validations:
            print(f"🔍 验证 {validation_name}...")
            try:
                result = validation_func()
                self.test_results[validation_name] = result

                if result['status'] == 'passed':
                    print(f"  ✅ {validation_name} 通过")
                    if 'details' in result:
                        for key, value in result['details'].items():
                            print(f"     {key}: {value}")
                else:
                    print(f"  ❌ {validation_name} 失败: {result.get('message', 'Unknown error')}")
                    all_passed = False

            except Exception as e:
                print(f"  ❌ {validation_name} 异常: {str(e)}")
                self.test_results[validation_name] = {'status': 'error', 'message': str(e)}
                all_passed = False

            print()

        self.generate_validation_report(all_passed)
        return all_passed

    def validate_cache_tests(self):
        """验证缓存系统测试"""
        print("  运行缓存核心组件测试...")

        result = self.run_pytest_command([
            "tests/unit/infrastructure/cache/test_cache_core_components.py",
            "-v", "--tb=short", "--timeout=30", "--maxfail=1"
        ])

        if result['returncode'] == 0:
            return {
                'status': 'passed',
                'details': {
                    'tests_run': result.get('tests_collected', 0),
                    'tests_passed': result.get('tests_passed', 0),
                    'duration': result.get('duration', 0)
                }
            }
        else:
            return {
                'status': 'failed',
                'message': f'缓存测试失败: {result.get("stderr", "Unknown error")}'
            }

    def validate_concurrent_tests(self):
        """验证并发测试"""
        print("  验证并发测试修复效果...")

        concurrent_tests = [
            "tests/unit/infrastructure/cache/test_cache_system.py::TestCacheSystemIntegration::test_thread_safety",
            "tests/unit/infrastructure/test_boundary_conditions.py::TestConfigBoundaryConditions::test_config_concurrent_access",
            "tests/unit/infrastructure/cache/test_cache_system.py::TestMultiLevelCache::test_set_and_get"
        ]

        results = []
        for test_path in concurrent_tests:
            print(f"    测试 {test_path.split('::')[-1]}...")
            result = self.run_pytest_command([
                test_path, "-v", "--tb=line", "--timeout=40"
            ])

            if result['returncode'] == 0:
                results.append(True)
                print("      ✅ 通过")
            else:
                results.append(False)
                print(f"      ❌ 失败: {result.get('stderr', '')}")

        passed_count = sum(results)
        total_count = len(results)

        if passed_count == total_count:
            return {
                'status': 'passed',
                'details': {
                    'concurrent_tests_passed': f'{passed_count}/{total_count}',
                    'all_concurrent_tests_working': True
                }
            }
        else:
            return {
                'status': 'failed',
                'message': f'并发测试失败: {passed_count}/{total_count} 通过'
            }

    def validate_interface_definitions(self):
        """验证接口定义"""
        print("  检查接口定义完整性...")

        # 检查ICacheManager接口
        try:
            from src.infrastructure.cache.global_interfaces import ICacheManager
            import inspect

            # 检查必要的抽象方法
            abstract_methods = []
            for name, method in inspect.getmembers(ICacheManager, predicate=inspect.isfunction):
                if hasattr(method, '__isabstractmethod__') and method.__isabstractmethod__:
                    abstract_methods.append(name)

            required_methods = ['get', 'set', 'delete',
                                'exists', 'clear', 'get_stats', 'is_healthy']
            missing_methods = [m for m in required_methods if m not in abstract_methods]

            if not missing_methods:
                return {
                    'status': 'passed',
                    'details': {
                        'interface_defined': True,
                        'abstract_methods': len(abstract_methods),
                        'required_methods_present': True
                    }
                }
            else:
                return {
                    'status': 'failed',
                    'message': f'缺少必要的抽象方法: {missing_methods}'
                }

        except ImportError as e:
            return {
                'status': 'failed',
                'message': f'接口导入失败: {e}'
            }

    def validate_timeout_mechanisms(self):
        """验证超时机制"""
        print("  检查超时机制配置...")

        # 检查pytest.ini中的超时配置
        pytest_ini = self.project_root / "pytest.ini"
        if not pytest_ini.exists():
            return {
                'status': 'warning',
                'message': 'pytest.ini文件不存在'
            }

        with open(pytest_ini, 'r') as f:
            content = f.read()

        # 检查是否配置了超时插件
        if 'timeout' in content:
            return {
                'status': 'passed',
                'details': {
                    'pytest_timeout_configured': True,
                    'timeout_plugin_enabled': True
                }
            }
        else:
            return {
                'status': 'warning',
                'message': '未在pytest.ini中发现超时配置'
            }

    def validate_full_test_suite(self):
        """验证完整测试套件"""
        print("  运行基础设施层完整测试套件...")

        result = self.run_pytest_command([
            "tests/unit/infrastructure/",
            "--tb=line",
            "--timeout=60",
            "--maxfail=5",
            "-q",
            "--disable-warnings"
        ])

        total_tests = result.get('tests_collected', 0)
        passed_tests = result.get('tests_passed', 0)

        if result['returncode'] == 0:
            return {
                'status': 'passed',
                'details': {
                    'total_infrastructure_tests': total_tests,
                    'tests_passed': passed_tests,
                    'test_success_rate': ".1f",
                    'full_suite_execution': 'successful'
                }
            }
        else:
            failed_tests = total_tests - passed_tests
            return {
                'status': 'warning' if failed_tests <= 5 else 'failed',
                'message': f'完整测试套件运行结果: {passed_tests}/{total_tests} 通过',
                'details': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests
                }
            }

    def run_pytest_command(self, args):
        """运行pytest命令"""
        cmd = [sys.executable, "-m", "pytest"] + args

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )
        except subprocess.TimeoutExpired:
            return {
                'returncode': -1,
                'stderr': 'Command timed out after 120 seconds',
                'duration': 120
            }

        duration = time.time() - start_time

        # 解析输出
        output = result.stdout + result.stderr

        # 提取测试统计信息
        tests_collected = 0
        tests_passed = 0

        for line in output.split('\n'):
            if 'collected' in line and 'items' in line:
                try:
                    tests_collected = int(line.split()[1])
                except:
                    pass
            elif 'passed' in line and 'failed' not in line:
                try:
                    tests_passed = int(line.split()[0])
                except:
                    pass

        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': duration,
            'tests_collected': tests_collected,
            'tests_passed': tests_passed
        }

    def generate_validation_report(self, all_passed):
        """生成验证报告"""
        print("="*60)
        print("📊 基础设施层测试验证报告")
        print("="*60)

        print(f"\n🔍 验证时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 项目路径: {self.project_root}")

        # 统计结果
        total_validations = len(self.test_results)
        passed_validations = sum(1 for r in self.test_results.values() if r['status'] == 'passed')
        warning_validations = sum(1 for r in self.test_results.values() if r['status'] == 'warning')
        failed_validations = sum(1 for r in self.test_results.values()
                                 if r['status'] != 'passed' and r['status'] != 'warning')

        print("\n📈 验证统计:")
        print(f"   总验证项: {total_validations}")
        print(f"   ✅ 通过: {passed_validations}")
        print(f"   ⚠️  警告: {warning_validations}")
        print(f"   ❌ 失败: {failed_validations}")
        print(".1f")
        # 详细结果
        if warning_validations > 0 or failed_validations > 0:
            print("\n⚠️  需要注意的项目:")
            for validation_name, result in self.test_results.items():
                if result['status'] != 'passed':
                    status_icon = "⚠️" if result['status'] == 'warning' else "❌"
                    print(f"   {status_icon} {validation_name}: {result.get('message', '需要检查')}")

        # 验证结论
        if all_passed:
            print("\n🎉 验证结论: 完全通过")
            print("   ✅ 所有基础设施层测试修复验证通过")
            print("   ✅ 系统现已稳定可靠")
            print("   ✅ 可以正常运行大规模测试套件")
        elif passed_validations >= total_validations * 0.8:
            print("\n🟡 验证结论: 基本通过")
            print("   ⚠️  大部分验证通过，少数项目需要注意")
            print("   ✅ 核心功能正常，系统基本稳定")
            print("   📋 建议处理警告项目后继续使用")
        else:
            print("\n🔴 验证结论: 需要改进")
            print("   ❌  多个关键验证项目失败")
            print("   📋 建议先解决失败的项目")

        print("\n🚀 建议下一步行动:")
        if all_passed:
            print("   1. ✅ 可以开始大规模测试运行")
            print("   2. 📊 监控测试执行时间和稳定性")
            print("   3. 🔄 应用这些修复模式到其他测试")
            print("   4. 📚 更新测试最佳实践文档")
        else:
            print("   1. 🔧 修复上述失败的项目")
            print("   2. 🧪 重新运行验证脚本")
            print("   3. 📊 分析失败原因和解决方案")
            print("   4. 📝 更新修复记录")

        print("\n" + "="*60)


def main():
    """主函数"""
    validator = InfrastructureTestValidator()
    success = validator.run_validation()

    if success:
        print("\n🎯 基础设施层测试验证全部通过!")
        print("系统现已稳定可靠，可以放心运行测试套件。")
    else:
        print("\n⚠️  发现需要改进的项目，请根据上述建议处理。")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

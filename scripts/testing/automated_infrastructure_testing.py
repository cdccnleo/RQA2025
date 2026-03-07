#!/usr/bin/env python3
"""
基础设施层自动化测试流程
集成线程清理管理器，确保测试前后的环境清洁
"""

from scripts.testing.thread_cleanup_manager import ThreadCleanupManager
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入线程清理管理器


class AutomatedInfrastructureTesting:
    """基础设施层自动化测试流程"""

    def __init__(self):
        self.project_root = project_root
        self.thread_manager = ThreadCleanupManager()
        self.test_results = {}

        # 定义测试模块和优先级
        self.test_modules = [
            {
                'name': 'monitoring',
                'path': 'tests/unit/infrastructure/monitoring',
                'priority': 'high',
                'status': 'ready',
                'description': '监控模块测试'
            },
            {
                'name': 'core',
                'path': 'tests/unit/infrastructure/core',
                'priority': 'high',
                'status': 'ready',
                'description': '核心模块测试'
            },
            {
                'name': 'performance',
                'path': 'tests/unit/infrastructure/performance',
                'priority': 'medium',
                'status': 'ready',
                'description': '性能模块测试'
            },
            {
                'name': 'error',
                'path': 'tests/unit/infrastructure/error',
                'priority': 'medium',
                'status': 'ready',
                'description': '错误处理模块测试'
            },
            {
                'name': 'resource',
                'path': 'tests/unit/infrastructure/resource',
                'priority': 'medium',
                'status': 'ready',
                'description': '资源管理模块测试'
            }
        ]

    def pre_test_cleanup(self) -> bool:
        """测试前清理"""
        print("🧹 测试前清理...")

        try:
            # 清理测试环境
            cleaned_count = self.thread_manager.cleanup_test_environment()
            print(f"✅ 清理了 {cleaned_count} 个线程")

            # 检查清理结果
            summary = self.thread_manager.get_thread_summary()
            if summary['total_threads'] <= 1:  # 只有主线程
                print("✅ 测试环境清理完成")
                return True
            else:
                print(f"⚠️  仍有 {summary['total_threads']} 个线程")
                return False

        except Exception as e:
            print(f"❌ 测试前清理失败: {e}")
            return False

    def post_test_cleanup(self) -> bool:
        """测试后清理"""
        print("🧹 测试后清理...")

        try:
            # 强制清理所有非主线程
            cleaned_count = self.thread_manager.cleanup_test_environment()
            print(f"✅ 清理了 {cleaned_count} 个线程")

            # 最终状态检查
            summary = self.thread_manager.get_thread_summary()
            print(f"最终线程数: {summary['total_threads']}")

            return summary['total_threads'] <= 1

        except Exception as e:
            print(f"❌ 测试后清理失败: {e}")
            return False

    def run_module_test(self, module: Dict) -> Dict:
        """运行单个模块的测试"""
        module_name = module['name']
        module_path = module['path']

        print(f"\n🧪 运行 {module_name} 模块测试...")
        print(f"   路径: {module_path}")
        print(f"   描述: {module['description']}")

        result = {
            'module': module_name,
            'success': False,
            'output': '',
            'error': '',
            'duration': 0,
            'threads_before': 0,
            'threads_after': 0
        }

        try:
            # 记录测试前线程数
            summary_before = self.thread_manager.get_thread_summary()
            result['threads_before'] = summary_before['total_threads']

            # 运行测试
            start_time = time.time()

            cmd = [
                sys.executable, "-m", "pytest",
                module_path,
                "-v", "--tb=short", "--timeout=30"
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2分钟超时
            )

            end_time = time.time()
            result['duration'] = end_time - start_time
            result['output'] = process.stdout
            result['error'] = process.stderr
            result['success'] = process.returncode == 0

            # 记录测试后线程数
            summary_after = self.thread_manager.get_thread_summary()
            result['threads_after'] = summary_after['total_threads']

            # 输出结果
            if result['success']:
                print(f"✅ {module_name} 模块测试通过")
                print(f"   耗时: {result['duration']:.2f}秒")
                print(f"   线程变化: {result['threads_before']} -> {result['threads_after']}")
            else:
                print(f"❌ {module_name} 模块测试失败")
                print(f"   耗时: {result['duration']:.2f}秒")
                print(f"   线程变化: {result['threads_before']} -> {result['threads_after']}")

            return result

        except subprocess.TimeoutExpired:
            result['error'] = "测试超时"
            print(f"⏰ {module_name} 模块测试超时")
            return result
        except Exception as e:
            result['error'] = str(e)
            print(f"❌ {module_name} 模块测试异常: {e}")
            return result

    def run_all_tests(self) -> Dict:
        """运行所有测试"""
        print("=" * 60)
        print("基础设施层自动化测试流程")
        print("=" * 60)

        all_results = {}

        try:
            # 1. 测试前清理
            if not self.pre_test_cleanup():
                print("❌ 测试前清理失败，终止测试")
                return all_results

            # 2. 按优先级运行测试
            for module in self.test_modules:
                if module['status'] == 'ready':
                    result = self.run_module_test(module)
                    all_results[module['name']] = result

                    # 测试后立即清理
                    if not self.post_test_cleanup():
                        print(f"⚠️  {module['name']} 模块测试后清理失败")

                    # 短暂休息
                    time.sleep(1)

            # 3. 最终清理
            print("\n🧹 最终清理...")
            final_cleanup = self.post_test_cleanup()

            # 4. 生成测试报告
            self.generate_test_report(all_results, final_cleanup)

            return all_results

        except Exception as e:
            print(f"❌ 自动化测试流程失败: {e}")
            return all_results

    def generate_test_report(self, results: Dict, final_cleanup: bool):
        """生成测试报告"""
        report_file = self.project_root / "reports" / "automated_infrastructure_testing_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 基础设施层自动化测试报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 测试概览\n\n")

            total_modules = len(results)
            successful_modules = sum(1 for r in results.values() if r['success'])
            failed_modules = total_modules - successful_modules

            f.write(f"- **总模块数**: {total_modules}\n")
            f.write(f"- **成功模块**: {successful_modules}\n")
            f.write(f"- **失败模块**: {failed_modules}\n")
            f.write(f"- **成功率**: {(successful_modules/total_modules*100):.1f}%\n")
            f.write(f"- **最终清理**: {'✅ 成功' if final_cleanup else '❌ 失败'}\n\n")

            f.write("## 详细结果\n\n")

            for module_name, result in results.items():
                status_icon = "✅" if result['success'] else "❌"
                f.write(f"### {status_icon} {module_name}\n\n")
                f.write(f"- **状态**: {'通过' if result['success'] else '失败'}\n")
                f.write(f"- **耗时**: {result['duration']:.2f}秒\n")
                f.write(f"- **线程变化**: {result['threads_before']} -> {result['threads_after']}\n")

                if result['error']:
                    f.write(f"- **错误**: {result['error']}\n")

                f.write("\n")

            f.write("## 总结\n\n")

            if successful_modules == total_modules:
                f.write("🎉 所有模块测试通过！基础设施层测试稳定性良好。\n\n")
            else:
                f.write(f"⚠️  有 {failed_modules} 个模块测试失败，需要进一步调查。\n\n")

            f.write("### 改进建议\n\n")
            f.write("1. **线程管理**: 继续优化线程退出机制\n")
            f.write("2. **测试隔离**: 确保每个测试用例的独立性\n")
            f.write("3. **超时控制**: 合理设置测试超时时间\n")
            f.write("4. **资源清理**: 测试前后的自动资源清理\n")

        print(f"📄 测试报告已生成: {report_file}")


def main():
    """主函数"""
    tester = AutomatedInfrastructureTesting()

    try:
        results = tester.run_all_tests()

        # 统计结果
        total = len(results)
        successful = sum(1 for r in results.values() if r['success'])

        print(f"\n🎯 测试完成!")
        print(f"   总模块: {total}")
        print(f"   成功: {successful}")
        print(f"   失败: {total - successful}")
        print(f"   成功率: {(successful/total*100):.1f}%")

        return 0 if successful == total else 1

    except KeyboardInterrupt:
        print("\n\n用户中断")
        return 1
    except Exception as e:
        print(f"\n\n自动化测试异常: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

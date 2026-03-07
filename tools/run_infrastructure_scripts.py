#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层脚本运行器
统一管理所有基础设施层脚本的执行
"""

import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import argparse


class InfrastructureScriptRunner:
    """基础设施层脚本运行器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "scripts" / "infrastructure"
        self.results = {
            'executed_scripts': [],
            'success_count': 0,
            'failure_count': 0,
            'total_time': 0
        }

        # 定义可用的脚本
        self.available_scripts = {
            'audit': {
                'file': 'audit_infrastructure_modules.py',
                'description': '审查基础设施层模块',
                'args': ['--project-root', str(self.project_root)]
            },
            'optimize': {
                'file': 'optimize_infrastructure.py',
                'description': '优化基础设施层代码',
                'args': ['--project-root', str(self.project_root)]
            },
            'fix-logging': {
                'file': 'fix_logging_dependencies.py',
                'description': '修复日志依赖',
                'args': ['--project-root', str(self.project_root)]
            },
            'test-logging': {
                'file': 'test_infrastructure_logging.py',
                'description': '测试基础设施层日志',
                'args': ['--project-root', str(self.project_root)]
            }
        }

    def run_script(self, script_name: str, additional_args: List[str] = None) -> Dict:
        """运行单个脚本"""
        if script_name not in self.available_scripts:
            return {
                'success': False,
                'error': f'脚本 {script_name} 不存在',
                'output': '',
                'duration': 0
            }

        script_info = self.available_scripts[script_name]
        script_path = self.scripts_dir / script_info['file']

        if not script_path.exists():
            return {
                'success': False,
                'error': f'脚本文件不存在: {script_path}',
                'output': '',
                'duration': 0
            }

        # 构建命令
        cmd = [sys.executable, str(script_path)] + script_info['args']
        if additional_args:
            cmd.extend(additional_args)

        try:
            start_time = time.time()

            # 运行脚本
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )

            duration = time.time() - start_time

            return {
                'success': result.returncode == 0,
                'output': result.stdout,
                'error': result.stderr if result.returncode != 0 else '',
                'duration': duration,
                'return_code': result.returncode
            }

        except Exception as e:
            return {
                'success': False,
                'error': f'运行脚本时发生异常: {str(e)}',
                'output': '',
                'duration': 0
            }

    def run_audit_workflow(self, generate_report: bool = True) -> Dict:
        """运行审查工作流程"""
        print("🚀 开始基础设施层审查工作流程...")
        print("=" * 50)

        workflow_results = {
            'audit': None,
            'optimize': None,
            'fix_logging': None,
            'test_logging': None
        }

        # 1. 运行模块审查
        print("📋 步骤 1: 运行模块审查...")
        audit_result = self.run_script('audit')
        workflow_results['audit'] = audit_result

        if audit_result['success']:
            print("✅ 模块审查完成")
            if audit_result['output']:
                print(audit_result['output'])
        else:
            print(f"❌ 模块审查失败: {audit_result['error']}")

        print()

        # 2. 运行优化脚本
        print("🔧 步骤 2: 运行优化脚本...")
        optimize_result = self.run_script('optimize')
        workflow_results['optimize'] = optimize_result

        if optimize_result['success']:
            print("✅ 优化脚本完成")
            if optimize_result['output']:
                print(optimize_result['output'])
        else:
            print(f"❌ 优化脚本失败: {optimize_result['error']}")

        print()

        # 3. 修复日志依赖
        print("🔧 步骤 3: 修复日志依赖...")
        fix_logging_result = self.run_script('fix-logging')
        workflow_results['fix_logging'] = fix_logging_result

        if fix_logging_result['success']:
            print("✅ 日志依赖修复完成")
            if fix_logging_result['output']:
                print(fix_logging_result['output'])
        else:
            print(f"❌ 日志依赖修复失败: {fix_logging_result['error']}")

        print()

        # 4. 测试日志功能
        print("🧪 步骤 4: 测试日志功能...")
        test_logging_result = self.run_script('test-logging')
        workflow_results['test_logging'] = test_logging_result

        if test_logging_result['success']:
            print("✅ 日志功能测试完成")
            if test_logging_result['output']:
                print(test_logging_result['output'])
        else:
            print(f"❌ 日志功能测试失败: {test_logging_result['error']}")

        print()

        # 生成工作流程报告
        if generate_report:
            report = self.generate_workflow_report(workflow_results)
            print("📄 工作流程报告:")
            print("=" * 50)
            print(report)

        return workflow_results

    def run_quick_check(self) -> Dict:
        """运行快速检查"""
        print("⚡ 运行基础设施层快速检查...")
        print("=" * 30)

        # 只运行审查和测试
        results = {
            'audit': self.run_script('audit'),
            'test_logging': self.run_script('test-logging')
        }

        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(results)

        print(f"✅ 快速检查完成: {success_count}/{total_count} 项通过")

        return results

    def run_full_optimization(self) -> Dict:
        """运行完整优化"""
        print("🔧 运行基础设施层完整优化...")
        print("=" * 30)

        # 运行所有优化相关脚本
        results = {
            'audit': self.run_script('audit'),
            'optimize': self.run_script('optimize'),
            'fix_logging': self.run_script('fix-logging'),
            'test_logging': self.run_script('test-logging')
        }

        success_count = sum(1 for r in results.values() if r['success'])
        total_count = len(results)

        print(f"✅ 完整优化完成: {success_count}/{total_count} 项通过")

        return results

    def list_available_scripts(self) -> None:
        """列出可用的脚本"""
        print("📋 可用的基础设施层脚本:")
        print("=" * 40)

        for name, info in self.available_scripts.items():
            script_path = self.scripts_dir / info['file']
            status = "✅ 可用" if script_path.exists() else "❌ 缺失"
            print(f"{name:15} - {info['description']} ({status})")

        print()
        print("使用方法:")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py <script_name>")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --workflow")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --quick-check")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --full-optimization")

    def generate_workflow_report(self, results: Dict) -> str:
        """生成工作流程报告"""
        report = []
        report.append("# 基础设施层工作流程报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 统计结果
        success_count = sum(1 for r in results.values() if r and r['success'])
        total_count = len([r for r in results.values() if r])

        report.append("## 执行统计")
        report.append(f"- 总步骤数: {total_count}")
        report.append(f"- 成功步骤数: {success_count}")
        report.append(f"- 失败步骤数: {total_count - success_count}")
        report.append(
            f"- 成功率: {success_count/total_count*100:.1f}%" if total_count > 0 else "- 成功率: 0%")
        report.append("")

        # 详细结果
        step_names = {
            'audit': '模块审查',
            'optimize': '代码优化',
            'fix_logging': '日志依赖修复',
            'test_logging': '日志功能测试'
        }

        report.append("## 详细结果")
        for step, result in results.items():
            if result:
                step_name = step_names.get(step, step)
                status = "✅ 成功" if result['success'] else "❌ 失败"
                duration = f"{result['duration']:.2f}秒"

                report.append(f"### {step_name}")
                report.append(f"- 状态: {status}")
                report.append(f"- 耗时: {duration}")

                if result['error']:
                    report.append(f"- 错误: {result['error']}")

                if result['output']:
                    # 只显示输出的前几行
                    output_lines = result['output'].strip().split('\n')[:10]
                    report.append(f"- 输出预览:")
                    for line in output_lines:
                        report.append(f"  {line}")
                    if len(result['output'].split('\n')) > 10:
                        report.append("  ...")

                report.append("")

        # 建议
        report.append("## 建议")
        if success_count == total_count:
            report.append("✅ 所有步骤都成功完成，基础设施层状态良好！")
        else:
            report.append("❌ 部分步骤失败，需要进一步检查和修复")

        report.append("1. 检查失败的步骤")
        report.append("2. 查看详细的错误信息")
        report.append("3. 手动修复发现的问题")
        report.append("4. 重新运行工作流程验证修复效果")

        return "\n".join(report)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基础设施层脚本运行器")
    parser.add_argument("script", nargs="?", help="要运行的脚本名称")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--workflow", action="store_true", help="运行完整工作流程")
    parser.add_argument("--quick-check", action="store_true", help="运行快速检查")
    parser.add_argument("--full-optimization", action="store_true", help="运行完整优化")
    parser.add_argument("--list", action="store_true", help="列出可用的脚本")
    parser.add_argument("--args", nargs="*", help="传递给脚本的额外参数")

    args = parser.parse_args()

    # 创建运行器
    runner = InfrastructureScriptRunner(args.project_root)

    if args.list:
        runner.list_available_scripts()
        return

    if args.workflow:
        runner.run_audit_workflow()
        return

    if args.quick_check:
        runner.run_quick_check()
        return

    if args.full_optimization:
        runner.run_full_optimization()
        return

    if args.script:
        # 运行单个脚本
        print(f"🚀 运行脚本: {args.script}")
        result = runner.run_script(args.script, args.args)

        if result['success']:
            print("✅ 脚本执行成功")
            if result['output']:
                print(result['output'])
        else:
            print(f"❌ 脚本执行失败: {result['error']}")
            if result['output']:
                print("输出:")
                print(result['output'])
    else:
        # 显示帮助信息
        print("🔧 基础设施层脚本运行器")
        print("=" * 30)
        print("使用方法:")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py <script_name>")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --workflow")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --quick-check")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --full-optimization")
        print("  python scripts/infrastructure/run_infrastructure_scripts.py --list")
        print()
        print("使用 --list 查看所有可用的脚本")


if __name__ == "__main__":
    main()

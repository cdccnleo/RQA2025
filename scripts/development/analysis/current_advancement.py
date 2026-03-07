#!/usr/bin/env python3
"""
RQA2025 当前模型落地推进脚本

专门处理当前的模型落地问题，包括：
1. 修复导入错误
2. 运行测试
3. 生成报告
"""

import subprocess
import time
from pathlib import Path
from datetime import datetime
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CurrentAdvancement:
    """当前推进器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.layers = [
            'infrastructure',
            'data',
            'features',
            'models',
            'trading',
            'backtest'
        ]

    def run_layer_tests(self, layer_name: str) -> dict:
        """运行指定层的测试"""
        print(f"\n🔧 开始测试 {layer_name.upper()} 层...")

        test_path = f"tests/unit/{layer_name}/"
        src_path = f"src/{layer_name}"

        # 检查路径是否存在
        if not (self.project_root / test_path).exists():
            print(f"❌ 测试路径不存在: {test_path}")
            return {
                'layer': layer_name,
                'status': 'failed',
                'error': 'Test path not found',
                'coverage': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            }

        # 运行测试
        cmd = [
            'conda', 'run', '-n', 'rqa', 'python', '-m', 'pytest',
            test_path,
            f'--cov={src_path}',
            '--cov-report=term-missing',
            '--cov-report=html',
            '-v',
            '--timeout=300',
            '--maxfail=10'
        ]

        print(f"📋 执行命令: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )
            end_time = time.time()

            # 解析结果
            output = result.stdout + result.stderr

            # 提取覆盖率
            coverage = 0
            if 'TOTAL' in output:
                for line in output.split('\n'):
                    if 'TOTAL' in line and '%' in line:
                        try:
                            coverage = float(line.split('%')[0].split()[-1])
                            break
                        except:
                            pass

            # 统计测试结果
            passed = output.count('PASSED')
            failed = output.count('FAILED')
            errors = output.count('ERROR')

            status = 'passed' if result.returncode == 0 else 'failed'

            print(f"✅ {layer_name.upper()} 层测试完成")
            print(f"   - 状态: {status}")
            print(f"   - 覆盖率: {coverage}%")
            print(f"   - 通过: {passed}, 失败: {failed}, 错误: {errors}")
            print(f"   - 耗时: {end_time - start_time:.2f}秒")

            return {
                'layer': layer_name,
                'status': status,
                'coverage': coverage,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'output': output
            }

        except subprocess.TimeoutExpired:
            print(f"❌ {layer_name.upper()} 层测试超时")
            return {
                'layer': layer_name,
                'status': 'timeout',
                'coverage': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            }
        except Exception as e:
            print(f"❌ {layer_name.upper()} 层测试异常: {e}")
            return {
                'layer': layer_name,
                'status': 'error',
                'error': str(e),
                'coverage': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0
            }

    def fix_import_issues(self):
        """修复导入问题"""
        print("\n🔧 修复导入问题...")

        # 运行模块修复脚本
        try:
            result = subprocess.run(
                ['python', 'scripts/comprehensive_module_fixer.py'],
                capture_output=True,
                text=True,
                timeout=300
            )
            print("✅ 模块修复完成")
            print(result.stdout)
        except Exception as e:
            print(f"❌ 模块修复失败: {e}")

    def run_all_layers(self):
        """运行所有层"""
        print("🚀 开始当前模型落地推进...")

        # 先修复导入问题
        self.fix_import_issues()

        results = []

        for layer in self.layers:
            result = self.run_layer_tests(layer)
            results.append(result)

            # 短暂休息
            time.sleep(2)

        # 生成报告
        self.generate_report(results)

    def generate_report(self, results):
        """生成报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"reports/current_advancement_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 当前模型落地推进报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 执行概览\n")
            f.write(f"- **总层数**: {len(results)}\n")
            f.write(f"- **成功层数**: {len([r for r in results if r['status'] == 'passed'])}\n")
            f.write(f"- **失败层数**: {len([r for r in results if r['status'] != 'passed'])}\n\n")

            f.write("## 详细结果\n\n")
            for result in results:
                f.write(f"### {result['layer'].upper()} 层\n")
                f.write(f"- **状态**: {result['status']}\n")
                f.write(f"- **覆盖率**: {result['coverage']}%\n")
                f.write(f"- **通过**: {result['passed']}\n")
                f.write(f"- **失败**: {result['failed']}\n")
                f.write(f"- **错误**: {result['errors']}\n")
                if 'error' in result:
                    f.write(f"- **错误信息**: {result['error']}\n")
                f.write("\n")

        print(f"📄 报告已保存到: {report_file}")


def main():
    """主函数"""
    advancement = CurrentAdvancement()
    advancement.run_all_layers()


if __name__ == "__main__":
    main()

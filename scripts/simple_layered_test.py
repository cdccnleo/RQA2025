#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 简化分层测试执行器

避免编码问题，简单地运行各层的测试
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

class SimpleLayeredTestRunner:
    """简化分层测试运行器"""

    def __init__(self):
        self.test_layers = {
            'infrastructure': {
                'name': '基础设施层',
                'path': 'tests/unit/infrastructure/',
                'target_coverage': 95,
                'description': '配置、缓存、日志、安全、错误处理、资源管理'
            },
            'data': {
                'name': '数据层',
                'path': 'tests/unit/data/',
                'target_coverage': 95,
                'description': '数据适配器、加载器、处理器、验证器'
            },
            'features': {
                'name': '特征层',
                'path': 'tests/unit/features/',
                'target_coverage': 90,
                'description': '特征工程、处理器、加速器、监控器'
            },
            'ml': {
                'name': '模型层',
                'path': 'tests/unit/ml/',
                'target_coverage': 90,
                'description': '模型管理器、推理引擎、集成器'
            },
            'core': {
                'name': '核心层',
                'path': 'tests/unit/core/',
                'target_coverage': 98,
                'description': '业务流程编排器、事件总线、服务容器'
            },
            'risk': {
                'name': '风控层',
                'path': 'tests/unit/risk/',
                'target_coverage': 95,
                'description': '风险管理器、合规检查器、告警系统'
            },
            'trading': {
                'name': '交易层',
                'path': 'tests/unit/trading/',
                'target_coverage': 90,
                'description': '交易引擎、订单管理器、执行引擎'
            },
            'engine': {
                'name': '引擎层',
                'path': 'tests/unit/engine/',
                'target_coverage': 95,
                'description': '实时引擎、性能监控器、系统监控器'
            }
        }

    def run_layered_tests(self):
        """运行分层测试"""
        print("🏗️ RQA2025 简化分层测试执行")
        print("=" * 60)

        results = {}

        for layer_key, layer_config in self.test_layers.items():
            print(f"\n🔬 执行 {layer_config['name']} 测试")
            print("-" * 50)
            print(f"📁 测试路径: {layer_config['path']}")
            print(f"🎯 目标覆盖率: {layer_config['target_coverage']}%")

            result = self._run_single_layer_test(layer_config)
            results[layer_key] = result

            # 显示结果摘要
            if result['status'] == 'PASSED':
                print(f"✅ {layer_config['name']} - 通过")
            elif result['status'] == 'FAILED':
                print(f"❌ {layer_config['name']} - 失败")
            else:
                print(f"⚠️ {layer_config['name']} - {result['status']}")

            if result.get('test_count', 0) > 0:
                print(f"🧪 测试数: {result['test_count']}")

        # 生成汇总报告
        self._generate_summary_report(results)

    def _run_single_layer_test(self, layer_config):
        """运行单层测试"""
        test_path = layer_config['path']

        if not os.path.exists(test_path):
            return {
                'status': 'MISSING',
                'error': f'测试路径不存在: {test_path}',
                'test_count': 0
            }

        try:
            # 使用最简单的pytest命令
            cmd = [
                sys.executable, '-m', 'pytest',
                test_path,
                '--tb=short',
                '-q'
            ]

            print(f"执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=120  # 2分钟超时
            )

            # 解析结果
            if result.returncode == 0:
                status = 'PASSED'
            else:
                status = 'FAILED'

            # 粗略估算测试数量
            test_count = self._estimate_test_count(result.stdout)

            return {
                'status': status,
                'returncode': result.returncode,
                'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr,
                'test_count': test_count
            }

        except subprocess.TimeoutExpired:
            return {
                'status': 'TIMEOUT',
                'error': '测试执行超时',
                'test_count': 0
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': f'执行测试时发生错误: {e}',
                'test_count': 0
            }

    def _estimate_test_count(self, stdout):
        """粗略估算测试数量"""
        # 从输出中查找测试数量信息
        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'passed' in line.lower() or 'failed' in line.lower():
                # 尝试解析类似 "5 passed, 2 failed"
                import re
                numbers = re.findall(r'\d+', line)
                if numbers:
                    return sum(int(n) for n in numbers)
        return 0

    def _generate_summary_report(self, results):
        """生成汇总报告"""
        print(f"\n{'=' * 80}")
        print("🏗️ RQA2025 分层测试执行汇总报告")
        print(f"{'=' * 80}")

        passed_count = sum(1 for r in results.values() if r.get('status') == 'PASSED')
        failed_count = sum(1 for r in results.values() if r.get('status') == 'FAILED')
        total_count = len(results)

        print(f"📊 执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"✅ 通过层数: {passed_count}/{total_count}")
        print(f"❌ 失败层数: {failed_count}/{total_count}")

        if passed_count == total_count:
            print("🎉 所有层级测试通过！")
        else:
            print("\n🔧 需要修复的层级:")
            for layer_key, result in results.items():
                if result.get('status') != 'PASSED':
                    layer_name = self.test_layers[layer_key]['name']
                    print(f"   • {layer_name}: {result.get('status')}")

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/SIMPLE_LAYERED_TEST_{timestamp}.json"

        report = {
            'simple_layered_test': {
                'execution_date': datetime.now().isoformat(),
                'results': results,
                'summary': {
                    'total_layers': total_count,
                    'passed_layers': passed_count,
                    'failed_layers': failed_count,
                    'success_rate': (passed_count / total_count * 100) if total_count > 0 else 0
                }
            }
        }

        os.makedirs('reports', exist_ok=True)
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n📄 详细报告已保存到: {report_file}")

def main():
    """主函数"""
    try:
        runner = SimpleLayeredTestRunner()
        runner.run_layered_tests()
        return 0
    except Exception as e:
        print(f"❌ 执行分层测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())


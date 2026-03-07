#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 分层测试执行器

按照生产测试计划分层分模块执行测试，生成分层测试报告
"""

import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LayeredTestExecutor:
    """分层测试执行器"""

    def __init__(self):
        self.layer_results = {}
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

    def run_layered_tests(self) -> Dict[str, Any]:
        """执行分层测试"""
        print("🏗️ RQA2025 分层测试执行")
        print("=" * 60)

        # 预处理：确保全局接口可用
        self._ensure_global_interfaces()

        for layer_key, layer_config in self.test_layers.items():
            print(f"\n🔬 执行 {layer_config['name']} 测试")
            print("-" * 50)
            print(f"📁 测试路径: {layer_config['path']}")
            print(f"🎯 目标覆盖率: {layer_config['target_coverage']}%")
            print(f"📋 职责: {layer_config['description']}")

            layer_result = self._run_single_layer_test(layer_key, layer_config)
            self.layer_results[layer_key] = layer_result

            # 显示结果摘要
            status = layer_result.get('status', 'UNKNOWN')
            status_emoji = {
                'PASSED': '✅',
                'FAILED': '❌',
                'WARNING': '⚠️',
                'ERROR': '💥'
            }

            print(f"\n{status_emoji.get(status, '❓')} {layer_config['name']} - {status}")

            if 'coverage' in layer_result:
                coverage = layer_result['coverage']
                target = layer_config['target_coverage']
                print(f"📊 实际覆盖率: {coverage:.1f}% (目标: {target}%)")
        return self._generate_comprehensive_report()

    def _run_single_layer_test(self, layer_key: str, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """执行单层测试"""
        test_path = layer_config['path']
        target_coverage = layer_config['target_coverage']

        if not os.path.exists(test_path):
            return {
                'status': 'ERROR',
                'error': f"测试路径不存在: {test_path}",
                'test_count': 0,
                'passed': 0,
                'failed': 0,
                'coverage': 0
            }

        try:
            # 使用pytest-cov运行测试并收集覆盖率
            cmd = [
                sys.executable, '-m', 'pytest', test_path,
                '--cov=src', f'--cov-report=term-missing',
                f'--cov-report=html:reports/coverage_{layer_key}',
                '--tb=no', '-q'
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # 解析测试结果
            test_result = self._parse_test_output(result.stdout, result.stderr, result.returncode)

            # 计算覆盖率
            coverage = self._extract_coverage_from_output(result.stdout)

            # 评估状态
            if test_result['returncode'] == 0:
                if coverage >= target_coverage:
                    status = 'PASSED'
                else:
                    status = 'WARNING'
            else:
                status = 'FAILED'

            return {
                'status': status,
                'test_path': test_path,
                'returncode': test_result['returncode'],
                'test_count': test_result['test_count'],
                'passed': test_result['passed'],
                'failed': test_result['failed'],
                'errors': test_result['errors'],
                'coverage': coverage,
                'target_coverage': target_coverage,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                'status': 'ERROR',
                'error': '测试执行超时',
                'timeout': True
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': f'执行测试时发生错误: {e}'
            }

    def _parse_test_output(self, stdout: str, stderr: str, returncode: int) -> Dict[str, Any]:
        """解析pytest输出"""
        result = {
            'returncode': returncode,
            'test_count': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'warnings': 0
        }

        # 解析测试结果行 (通常在最后)
        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'passed' in line and 'failed' in line:
                # 解析类似 "5 passed, 2 failed, 1 error in 10.23s"
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if 'passed' in part:
                        try:
                            result['passed'] = int(part.split()[0])
                        except:
                            pass
                    elif 'failed' in part:
                        try:
                            result['failed'] = int(part.split()[0])
                        except:
                            pass
                    elif 'error' in part:
                        try:
                            result['errors'] = int(part.split()[0])
                        except:
                            pass
                    elif 'warning' in part:
                        try:
                            result['warnings'] = int(part.split()[0])
                        except:
                            pass
                break

        result['test_count'] = result['passed'] + result['failed'] + result['errors']
        return result

    def _extract_coverage_from_output(self, stdout: str) -> float:
        """从pytest输出中提取覆盖率"""
        lines = stdout.split('\n')
        for line in reversed(lines):
            if 'TOTAL' in line and '%' in line:
                # 解析类似 "TOTAL                                85%   1250   220"
                parts = line.split()
                for part in parts:
                    if '%' in part:
                        try:
                            return float(part.replace('%', ''))
                        except:
                            pass
        return 0.0

    def _ensure_global_interfaces(self):
        """确保全局接口可用"""
        print("🔧 确保全局接口可用...")

        # 创建全局接口导入文件
        global_import_content = '''
"""
全局接口导入文件

确保所有测试中需要的接口都能被正确导入
"""

# 缓存相关接口
try:
    from src.infrastructure.cache.interfaces import ICacheStrategy, CacheEvictionStrategy
except ImportError:
    print("Warning: Could not import cache interfaces")

# 数据相关接口
try:
    from src.data.lake.partition_manager import PartitionStrategy
except ImportError:
    from enum import Enum
    class PartitionStrategy(Enum):
        DATE = "date"
        HASH = "hash"
        CUSTOM = "custom"
        RANGE = "range"

try:
    from src.data.repair.data_repairer import RepairStrategy
except ImportError:
    from enum import Enum
    class RepairStrategy(Enum):
        FILL_FORWARD = "fill_forward"
        FILL_BACKWARD = "fill_backward"
        FILL_MEAN = "fill_mean"
        FILL_MEDIAN = "fill_median"
        FILL_MODE = "fill_mode"
        REMOVE_OUTLIERS = "remove_outliers"
        DROP = "drop"
        LOG_TRANSFORM = "log_transform"
        INTERPOLATE = "interpolate"

# 导出到全局命名空间
import sys
current_module = sys.modules[__name__]

for name in ['ICacheStrategy', 'CacheEvictionStrategy', 'PartitionStrategy', 'RepairStrategy']:
    if name in locals():
        setattr(current_module, name, locals()[name])
'''

        os.makedirs('tests', exist_ok=True)
        with open('tests/global_imports.py', 'w', encoding='utf-8') as f:
            f.write(global_import_content)

        print("✅ 全局接口文件已创建")

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        # 计算总体统计
        total_layers = len(self.layer_results)
        passed_layers = sum(1 for r in self.layer_results.values() if r.get('status') == 'PASSED')
        failed_layers = sum(1 for r in self.layer_results.values() if r.get('status') == 'FAILED')
        warning_layers = sum(1 for r in self.layer_results.values() if r.get('status') == 'WARNING')
        error_layers = sum(1 for r in self.layer_results.values() if r.get('status') == 'ERROR')

        # 计算总体覆盖率
        valid_coverages = [r.get('coverage', 0)
                           for r in self.layer_results.values() if r.get('coverage', 0) > 0]
        avg_coverage = sum(valid_coverages) / len(valid_coverages) if valid_coverages else 0

        # 计算总体测试统计
        total_tests = sum(r.get('test_count', 0) for r in self.layer_results.values())
        total_passed = sum(r.get('passed', 0) for r in self.layer_results.values())
        total_failed = sum(r.get('failed', 0) for r in self.layer_results.values())

        report = {
            'layered_test_execution': {
                'project_name': 'RQA2025 量化交易系统',
                'execution_date': datetime.now().isoformat(),
                'version': '2.0',
                'layer_results': self.layer_results,
                'summary': {
                    'total_layers': total_layers,
                    'passed_layers': passed_layers,
                    'failed_layers': failed_layers,
                    'warning_layers': warning_layers,
                    'error_layers': error_layers,
                    'average_coverage': avg_coverage,
                    'total_tests': total_tests,
                    'total_passed': total_passed,
                    'total_failed': total_failed,
                    'overall_success_rate': (passed_layers / total_layers * 100) if total_layers > 0 else 0
                },
                'test_layers_config': self.test_layers,
                'recommendations': self._generate_layer_recommendations(),
                'generated_at': datetime.now().isoformat()
            }
        }

        return report

    def _generate_layer_recommendations(self) -> List[str]:
        """生成分层建议"""
        recommendations = []

        # 基于每层结果生成建议
        for layer_key, layer_config in self.test_layers.items():
            result = self.layer_results.get(layer_key, {})

            if result.get('status') == 'FAILED':
                recommendations.append(f"🔴 {layer_config['name']}: 修复测试失败问题")
            elif result.get('status') == 'WARNING':
                coverage = result.get('coverage', 0)
                target = layer_config['target_coverage']
                recommendations.append(
                    f"🟡 {layer_config['name']}: 覆盖率不足 - 实际: {coverage:.1f}%, 目标: {target}%"
                )
            elif result.get('status') == 'ERROR':
                recommendations.append(f"💥 {layer_config['name']}: 解决测试执行错误")
            else:
                recommendations.append(f"✅ {layer_config['name']}: 测试通过，可继续优化")

        return recommendations


def main():
    """主函数"""
    try:
        executor = LayeredTestExecutor()
        report = executor.run_layered_tests()

        # 保存分层测试报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/LAYERED_TEST_EXECUTION_{timestamp}.json"

        os.makedirs('reports', exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 打印摘要报告
        data = report['layered_test_execution']
        summary = data['summary']

        print(f"\n{'=' * 80}")
        print("🏗️ RQA2025 分层测试执行报告")
        print(f"{'=' * 80}")
        print(
            f"📅 执行日期: {datetime.fromisoformat(data['execution_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"📊 总体状态: {'SUCCESS' if summary['overall_success_rate'] >= 80 else 'NEEDS_IMPROVEMENT'}")
        print(f"✅ 通过层数: {summary['passed_layers']}/{summary['total_layers']}")
        print(f"📈 总体成功率: {summary['overall_success_rate']:.1f}%")
        print(f"📈 平均覆盖率: {summary['average_coverage']:.1f}%")
        print(f"🧪 总测试数: {summary['total_tests']}")
        print(f"✅ 总通过数: {summary['total_passed']}")
        print(f"❌ 总失败数: {summary['total_failed']}")

        print(f"\n📋 分层建议:")
        for rec in data.get('recommendations', []):
            print(f"   {rec}")

        print(f"\n📄 详细报告已保存到: {report_file}")

        # 返回状态码
        success_rate = summary['overall_success_rate']
        if success_rate >= 90:
            return 0  # 优秀
        elif success_rate >= 80:
            return 1  # 良好
        elif success_rate >= 60:
            return 2  # 需要改进
        else:
            return 3  # 严重问题

    except Exception as e:
        print(f"❌ 执行分层测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

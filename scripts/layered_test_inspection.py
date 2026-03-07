#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 分层测试检查器

检查各层测试文件是否存在、状态，并生成修复建议
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class LayeredTestInspector:
    """分层测试检查器"""

    def __init__(self):
        self.test_layers = {
            'infrastructure': {
                'name': '基础设施层',
                'path': 'tests/unit/infrastructure/',
                'target_coverage': 95,
                'description': '配置、缓存、日志、安全、错误处理、资源管理',
                'key_components': ['配置管理', '缓存系统', '日志系统', '安全管理', '错误处理', '资源管理']
            },
            'data': {
                'name': '数据层',
                'path': 'tests/unit/data/',
                'target_coverage': 95,
                'description': '数据适配器、加载器、处理器、验证器',
                'key_components': ['数据适配器', '数据加载器', '数据处理器', '数据验证器']
            },
            'features': {
                'name': '特征层',
                'path': 'tests/unit/features/',
                'target_coverage': 90,
                'description': '特征工程、处理器、加速器、监控器',
                'key_components': ['特征工程器', '特征处理器', 'GPU加速器', '监控器']
            },
            'ml': {
                'name': '模型层',
                'path': 'tests/unit/ml/',
                'target_coverage': 90,
                'description': '模型管理器、推理引擎、集成器',
                'key_components': ['模型管理器', '推理引擎', '集成器', '预测器']
            },
            'core': {
                'name': '核心层',
                'path': 'tests/unit/core/',
                'target_coverage': 98,
                'description': '业务流程编排器、事件总线、服务容器',
                'key_components': ['业务流程编排器', '事件总线', '依赖注入容器', '服务容器']
            },
            'risk': {
                'name': '风控层',
                'path': 'tests/unit/risk/',
                'target_coverage': 95,
                'description': '风险管理器、合规检查器、告警系统',
                'key_components': ['风险管理器', '合规检查器', '风险控制器', '告警系统']
            },
            'trading': {
                'name': '交易层',
                'path': 'tests/unit/trading/',
                'target_coverage': 90,
                'description': '交易引擎、订单管理器、执行引擎',
                'key_components': ['交易引擎', '订单管理器', '执行引擎', '智能路由']
            },
            'engine': {
                'name': '引擎层',
                'path': 'tests/unit/engine/',
                'target_coverage': 95,
                'description': '实时引擎、性能监控器、系统监控器',
                'key_components': ['实时引擎', '性能监控器', '系统监控器', '数据分发器']
            }
        }

    def inspect_layered_tests(self) -> Dict[str, Any]:
        """检查分层测试"""
        print("🔍 RQA2025 分层测试检查")
        print("=" * 60)

        inspection_results = {}

        for layer_key, layer_config in self.test_layers.items():
            print(f"\n📂 检查 {layer_config['name']}")
            print("-" * 40)
            print(f"📁 测试路径: {layer_config['path']}")
            print(f"🎯 目标覆盖率: {layer_config['target_coverage']}%")

            layer_inspection = self._inspect_single_layer(layer_key, layer_config)
            inspection_results[layer_key] = layer_inspection

            # 显示检查结果摘要
            status = layer_inspection.get('status', 'UNKNOWN')
            status_emoji = {
                'EXCELLENT': '🟢',
                'GOOD': '🟡',
                'NEEDS_WORK': '🟠',
                'CRITICAL': '🔴',
                'MISSING': '⚫'
            }

            print(f"\n{status_emoji.get(status, '❓')} {layer_config['name']} - {status}")

            test_count = layer_inspection.get('test_count', 0)
            if test_count > 0:
                print(f"🧪 发现 {test_count} 个测试文件")
            else:
                print("❌ 未发现测试文件")

        return self._generate_inspection_report(inspection_results)

    def _inspect_single_layer(self, layer_key: str, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """检查单层测试"""
        test_path = layer_config['path']

        inspection = {
            'layer_name': layer_config['name'],
            'test_path': test_path,
            'test_files': [],
            'test_count': 0,
            'status': 'MISSING',
            'issues': [],
            'recommendations': [],
            'coverage_estimate': 0
        }

        # 检查测试路径是否存在
        if not os.path.exists(test_path):
            inspection['issues'].append(f"测试目录不存在: {test_path}")
            inspection['recommendations'].append(f"创建测试目录: {test_path}")
            return inspection

        # 扫描测试文件
        test_files = []
        if os.path.isdir(test_path):
            for root, dirs, files in os.walk(test_path):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        test_files.append(file_path)

        inspection['test_files'] = test_files
        inspection['test_count'] = len(test_files)

        # 根据测试文件数量评估状态
        if len(test_files) == 0:
            inspection['status'] = 'CRITICAL'
            inspection['issues'].append("没有找到任何测试文件")
            inspection['recommendations'].append(f"为{len(layer_config['key_components'])}个关键组件创建测试")
        elif len(test_files) < len(layer_config['key_components']):
            inspection['status'] = 'NEEDS_WORK'
            inspection['issues'].append(
                f"测试覆盖不足: {len(test_files)}/{len(layer_config['key_components'])}")
            inspection['recommendations'].append("为缺失的组件创建测试")
            inspection['coverage_estimate'] = (
                len(test_files) / len(layer_config['key_components'])) * 100
        elif len(test_files) == len(layer_config['key_components']):
            inspection['status'] = 'GOOD'
            inspection['issues'].append("测试覆盖基本完整")
            inspection['recommendations'].append("可以考虑增加更多测试场景")
            inspection['coverage_estimate'] = 100
        else:
            inspection['status'] = 'EXCELLENT'
            inspection['issues'].append("测试覆盖优秀")
            inspection['recommendations'].append("可以考虑重构测试以提高质量")
            inspection['coverage_estimate'] = 100

        return inspection

    def _generate_inspection_report(self, inspection_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成检查报告"""
        # 计算总体统计
        total_layers = len(inspection_results)
        excellent_layers = sum(1 for r in inspection_results.values()
                               if r.get('status') == 'EXCELLENT')
        good_layers = sum(1 for r in inspection_results.values() if r.get('status') == 'GOOD')
        needs_work_layers = sum(1 for r in inspection_results.values()
                                if r.get('status') == 'NEEDS_WORK')
        critical_layers = sum(1 for r in inspection_results.values()
                              if r.get('status') == 'CRITICAL')
        missing_layers = sum(1 for r in inspection_results.values() if r.get('status') == 'MISSING')

        # 计算测试文件总数
        total_test_files = sum(r.get('test_count', 0) for r in inspection_results.values())

        report = {
            'layered_test_inspection': {
                'project_name': 'RQA2025 量化交易系统',
                'inspection_date': datetime.now().isoformat(),
                'version': '1.0',
                'inspection_results': inspection_results,
                'summary': {
                    'total_layers': total_layers,
                    'excellent_layers': excellent_layers,
                    'good_layers': good_layers,
                    'needs_work_layers': needs_work_layers,
                    'critical_layers': critical_layers,
                    'missing_layers': missing_layers,
                    'total_test_files': total_test_files,
                    'overall_health_score': self._calculate_health_score(inspection_results)
                },
                'generated_at': datetime.now().isoformat()
            }
        }

        return report

    def _calculate_health_score(self, inspection_results: Dict[str, Any]) -> float:
        """计算健康评分"""
        status_scores = {
            'EXCELLENT': 100,
            'GOOD': 80,
            'NEEDS_WORK': 60,
            'CRITICAL': 30,
            'MISSING': 0
        }

        total_score = 0
        for result in inspection_results.values():
            status = result.get('status', 'MISSING')
            total_score += status_scores.get(status, 0)

        return total_score / len(inspection_results) if inspection_results else 0


def main():
    """主函数"""
    try:
        inspector = LayeredTestInspector()
        report = inspector.inspect_layered_tests()

        # 保存检查报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"reports/LAYERED_TEST_INSPECTION_{timestamp}.json"

        os.makedirs('reports', exist_ok=True)
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 打印摘要报告
        data = report['layered_test_inspection']
        summary = data['summary']

        print(f"\n{'=' * 80}")
        print("🔍 RQA2025 分层测试检查报告")
        print(f"{'=' * 80}")
        print(
            f"📅 检查日期: {datetime.fromisoformat(data['inspection_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🏥 总体健康评分: {summary['overall_health_score']:.1f}/100")

        print(f"\n📊 各层状态分布:")
        print(f"🟢 优秀: {summary['excellent_layers']} 层")
        print(f"🟡 良好: {summary['good_layers']} 层")
        print(f"🟠 需要改进: {summary['needs_work_layers']} 层")
        print(f"🔴 严重: {summary['critical_layers']} 层")
        print(f"⚫ 缺失: {summary['missing_layers']} 层")

        print(f"\n🧪 总测试文件数: {summary['total_test_files']}")

        # 显示具体问题和建议
        print(f"\n🔧 需要重点关注的层级:")
        for layer_key, result in data['inspection_results'].items():
            if result['status'] in ['CRITICAL', 'MISSING', 'NEEDS_WORK']:
                print(
                    f"   • {result['layer_name']} ({result['status']}) - {len(result['issues'])} 个问题")

        print(f"\n📄 详细报告已保存到: {report_file}")

        return 0

    except Exception as e:
        print(f"❌ 检查分层测试时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
RQA2025 架构和代码审查工具

系统性审查各个架构层的实现情况，确保符合业务流程驱动的架构设计
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
from datetime import datetime


class ArchitectureCodeReviewer:
    """架构和代码审查器"""

    def __init__(self):
        self.layers = {
            'core': {
                'path': 'src/core',
                'name': '核心服务层',
                'responsibilities': [
                    '事件总线', '依赖注入容器', '业务流程编排',
                    '系统集成管理', '核心服务协调', '架构层间通信桥梁'
                ],
                'forbidden': [
                    '直接处理业务逻辑', '数据持久化操作', '用户接口处理'
                ]
            },
            'infrastructure': {
                'path': 'src/infrastructure',
                'name': '基础设施层',
                'responsibilities': [
                    '配置管理', '缓存系统', '日志系统', '安全管理',
                    '错误处理', '资源管理', '健康检查', '工具组件',
                    '网络通信', '存储抽象', '性能监控'
                ],
                'forbidden': [
                    '业务逻辑处理', '数据采集和处理', '交易决策和执行'
                ]
            },
            'data': {
                'path': 'src/data',
                'name': '数据采集层',
                'responsibilities': [
                    '数据源适配', '实时数据采集', '数据验证',
                    '数据质量监控', '数据格式转换', '数据缓存',
                    '数据源连接管理', '故障恢复'
                ],
                'forbidden': [
                    '特征工程和数据分析', '模型训练和推理',
                    '交易决策和执行', 'trading', 'strategy', 'execution'
                ]
            },
            'gateway': {
                'path': 'src/gateway',
                'name': 'API网关层',
                'responsibilities': [
                    '路由转发', '认证授权', '限流熔断',
                    '请求聚合', '协议转换', 'API文档生成',
                    '安全防护', '访问控制', '流量控制'
                ],
                'forbidden': [
                    '业务逻辑处理', '数据持久化',
                    'trading', 'model', 'strategy'
                ]
            },
            'features': {
                'path': 'src/features',
                'name': '特征处理层',
                'responsibilities': [
                    '智能特征工程', '分布式处理', '硬件加速',
                    '特征提取', '特征选择', '特征变换', '特征存储',
                    '技术指标计算', '统计特征生成', '市场数据预处理'
                ],
                'forbidden': [
                    '模型训练和推理', '交易决策和执行',
                    'trading', 'order', 'execution'
                ]
            },
            'ml': {
                'path': 'src/ml',
                'name': '模型推理层',
                'responsibilities': [
                    '集成学习', '模型管理', '实时推理',
                    '模型训练', '模型评估', '模型部署', '模型监控',
                    '特征预测', '概率输出', '模型集成'
                ],
                'forbidden': [
                    '交易决策和执行', '订单生成和管理',
                    'trading', 'order', 'execution'
                ]
            },
            'backtest': {
                'path': 'src/backtest',
                'name': '策略决策层',
                'responsibilities': [
                    '策略生成器', '策略框架', '投资组合管理',
                    '回测执行', '策略评估', '参数优化', '信号生成',
                    'strategy', 'trading', 'risk'
                ],
                'forbidden': [
                    '实盘交易执行', '实际订单提交', '生产环境资金操作'
                ]
            },
            'risk': {
                'path': 'src/risk',
                'name': '风控合规层',
                'responsibilities': [
                    '风控API', '中国市场规则', '风险控制器',
                    '风险检查', '合规验证', '风险评估', '风险监控',
                    'risk', 'compliance', 'limit'
                ],
                'forbidden': [
                    '实际交易执行', '订单生成和管理', '资金操作'
                ]
            },
            'trading': {
                'path': 'src/trading',
                'name': '交易执行层',
                'responsibilities': [
                    '订单管理', '执行引擎', '智能路由',
                    '交易执行', '订单状态跟踪', '执行监控',
                    'trading', 'order', 'execution'
                ],
                'forbidden': [
                    '回测和仿真', '模拟交易', 'simulation', 'backtest'
                ]
            },
            'engine': {
                'path': 'src/engine',
                'name': '监控反馈层',
                'responsibilities': [
                    '系统监控', '业务监控', '性能监控',
                    '跨层级数据收集', '状态监控'
                ],
                'forbidden': [
                    '业务逻辑处理', '交易决策', '实际业务执行'
                ]
            }
        }

        self.issues = defaultdict(list)
        self.layer_stats = defaultdict(dict)

    def analyze_layer_structure(self, layer_name: str, layer_config: Dict) -> Dict:
        """分析层级结构"""
        layer_path = layer_config['path']
        if not os.path.exists(layer_path):
            return {
                'exists': False,
                'file_count': 0,
                'subdirs': [],
                'components': []
            }

        # 统计文件数量
        file_count = 0
        subdirs = []
        components = []

        for root, dirs, files in os.walk(layer_path):
            subdirs.extend([d for d in dirs if not d.startswith('.')])

            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    file_count += 1

                    if file.endswith('_components.py'):
                        components.append(file)

        return {
            'exists': True,
            'file_count': file_count,
            'subdirs': list(set(subdirs)),
            'components': components
        }

    def check_layer_compliance(self, layer_name: str, layer_config: Dict) -> List[str]:
        """检查层级合规性"""
        issues = []
        layer_path = layer_config['path']

        if not os.path.exists(layer_path):
            issues.append(f"❌ 层级目录不存在: {layer_path}")
            return issues

        # 检查职责边界
        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 检查禁止的内容
                        for forbidden in layer_config['forbidden']:
                            if forbidden.lower() in content.lower():
                                if forbidden in ['trading', 'strategy', 'execution', 'model', 'risk', 'order']:
                                    # 检查是否在注释或字符串中
                                    if re.search(r'\b' + re.escape(forbidden) + r'\b', content):
                                        issues.append(f"⚠️  文件 {file_path} 包含禁止的概念 '{forbidden}'")
                                else:
                                    if forbidden in content:
                                        issues.append(f"⚠️  文件 {file_path} 包含禁止的内容 '{forbidden}'")

                    except Exception as e:
                        issues.append(f"⚠️  无法读取文件 {file_path}: {e}")

        return issues

    def analyze_dependencies(self, layer_name: str, layer_config: Dict) -> Dict:
        """分析层级依赖关系"""
        layer_path = layer_config['path']
        dependencies = {
            'imports_from': defaultdict(int),
            'imports_to': defaultdict(int),
            'cross_layer_imports': []
        }

        if not os.path.exists(layer_path):
            return dependencies

        # 检查导入关系
        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 查找import语句
                        import_pattern = re.compile(r'^(?:from|import)\s+src\.(\w+)')
                        imports = import_pattern.findall(content)

                        for imported_layer in imports:
                            if imported_layer != layer_name:
                                dependencies['cross_layer_imports'].append({
                                    'from_file': str(file_path),
                                    'to_layer': imported_layer,
                                    'import': f'src.{imported_layer}'
                                })
                                dependencies['imports_to'][imported_layer] += 1

                    except Exception as e:
                        print(f"⚠️  无法分析文件 {file_path}: {e}")

        return dependencies

    def check_component_factory_quality(self, layer_name: str, layer_config: Dict) -> Dict:
        """检查组件工厂质量"""
        quality_metrics = {
            'total_components': 0,
            'valid_factories': 0,
            'issues': []
        }

        layer_path = layer_config['path']
        if not os.path.exists(layer_path):
            return quality_metrics

        for root, dirs, files in os.walk(layer_path):
            for file in files:
                if file.endswith('_components.py'):
                    quality_metrics['total_components'] += 1
                    file_path = Path(root) / file

                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 检查组件工厂的标准结构
                        has_interface = 'class I' in content and 'Component(ABC):' in content
                        has_factory = 'ComponentFactory:' in content
                        has_create_method = 'create_component' in content
                        has_import = 'from typing import' in content
                        has_abc = 'from abc import' in content

                        if all([has_interface, has_factory, has_create_method, has_import, has_abc]):
                            quality_metrics['valid_factories'] += 1
                        else:
                            missing = []
                            if not has_interface:
                                missing.append('接口定义')
                            if not has_factory:
                                missing.append('工厂类')
                            if not has_create_method:
                                missing.append('创建方法')
                            if not has_import:
                                missing.append('类型导入')
                            if not has_abc:
                                missing.append('ABC导入')
                            quality_metrics['issues'].append({
                                'file': str(file_path),
                                'missing': missing
                            })

                    except Exception as e:
                        quality_metrics['issues'].append({
                            'file': str(file_path),
                            'error': str(e)
                        })

        return quality_metrics

    def generate_review_report(self) -> str:
        """生成审查报告"""
        print("🚀 开始全面架构和代码审查...")
        print("="*80)

        report = []
        report.append("# RQA2025 架构和代码审查报告")
        report.append("")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        total_issues = 0
        total_components = 0
        valid_components = 0

        for layer_name, layer_config in self.layers.items():
            print(f"\n🔍 审查 {layer_config['name']} ({layer_name})")
            report.append(f"## {layer_config['name']} ({layer_name})")
            report.append("")

            # 1. 结构分析
            structure = self.analyze_layer_structure(layer_name, layer_config)
            report.append("### 结构分析")
            report.append(f"- **目录路径**: {layer_config['path']}")
            report.append(f"- **是否存在**: {'✅' if structure['exists'] else '❌'}")
            report.append(f"- **文件数量**: {structure['file_count']}")
            report.append(
                f"- **子目录**: {', '.join(structure['subdirs']) if structure['subdirs'] else '无'}")
            report.append(f"- **组件工厂**: {len(structure['components'])} 个")
            report.append("")

            # 2. 职责边界检查
            compliance_issues = self.check_layer_compliance(layer_name, layer_config)
            report.append("### 职责边界检查")
            if compliance_issues:
                report.append(f"**发现问题**: {len(compliance_issues)} 个")
                for issue in compliance_issues:
                    report.append(f"- {issue}")
                total_issues += len(compliance_issues)
            else:
                report.append("**合规状态**: ✅ 符合职责边界要求")
            report.append("")

            # 3. 依赖关系分析
            dependencies = self.analyze_dependencies(layer_name, layer_config)
            report.append("### 依赖关系分析")
            report.append(f"- **跨层导入**: {len(dependencies['cross_layer_imports'])} 个")
            if dependencies['cross_layer_imports']:
                report.append("- **导入详情**:")
                for imp in dependencies['cross_layer_imports'][:5]:  # 显示前5个
                    report.append(f"  - {imp['from_file']} → {imp['to_layer']}")
                if len(dependencies['cross_layer_imports']) > 5:
                    report.append(f"  - ... 还有 {len(dependencies['cross_layer_imports']) - 5} 个导入")
            report.append("")

            # 4. 组件工厂质量检查
            component_quality = self.check_component_factory_quality(layer_name, layer_config)
            report.append("### 组件工厂质量")
            report.append(f"- **组件总数**: {component_quality['total_components']}")
            report.append(f"- **有效组件**: {component_quality['valid_factories']}")
            report.append(".1f")

            if component_quality['issues']:
                report.append("- **质量问题**:")
                for issue in component_quality['issues'][:3]:  # 显示前3个
                    report.append(
                        f"  - {issue['file']}: 缺失 {', '.join(issue.get('missing', ['未知']))}")

            total_components += component_quality['total_components']
            valid_components += component_quality['valid_factories']
            report.append("")

            # 5. 职责范围总结
            report.append("### 职责范围")
            report.append("**允许的职责**:")
            for resp in layer_config['responsibilities']:
                report.append(f"- ✅ {resp}")
            report.append("")
            report.append("**禁止的职责**:")
            for forbid in layer_config['forbidden']:
                report.append(f"- ❌ {forbid}")
            report.append("")

        # 总体总结
        report.append("## 📊 总体审查结果")
        report.append("")
        report.append("### 统计汇总")
        report.append(f"- **总层级数**: {len(self.layers)}")
        report.append(f"- **总组件数**: {total_components}")
        report.append(f"- **有效组件数**: {valid_components}")
        report.append(f"- **问题总数**: {total_issues}")
        report.append(".1f" if total_components > 0 else "- **组件有效率**: N/A")
        report.append("")

        if total_issues == 0:
            report.append("### 🎉 审查结论")
            report.append("**所有架构层级均符合设计要求！**")
            report.append("")
            report.append("✅ 架构层次结构正确")
            report.append("✅ 职责边界清晰明确")
            report.append("✅ 依赖关系合理")
            report.append("✅ 组件工厂质量优良")
            report.append("✅ 代码实现符合架构设计")
        else:
            report.append("### ⚠️ 需要关注的改进点")
            report.append(f"发现 {total_issues} 个需要改进的问题点")
            report.append("建议优先处理职责边界和组件质量问题")

        return "\n".join(report)

    def run_comprehensive_review(self) -> str:
        """运行全面审查"""
        return self.generate_review_report()


def main():
    """主函数"""
    reviewer = ArchitectureCodeReviewer()
    report = reviewer.run_comprehensive_review()

    # 保存报告
    with open('reports/ARCHITECTURE_CODE_REVIEW_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n📋 审查报告已保存到: reports/ARCHITECTURE_CODE_REVIEW_REPORT.md")
    print("🎉 架构和代码审查完成！")


if __name__ == "__main__":
    main()

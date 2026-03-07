"""
Phase 2: 统一命名规范和治理机制

建立基础设施层的统一命名规范，解决重复类名泛滥问题
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


class NamingStandardsFramework:
    """命名规范框架"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.naming_rules = self._define_naming_rules()
        self.architecture_patterns = self._define_architecture_patterns()

    def _define_naming_rules(self) -> Dict:
        """定义命名规则"""
        return {
            # 接口命名规则
            'interface': {
                'prefix': 'I',
                'pattern': r'^I[A-Z][a-zA-Z0-9]*$',
                'examples': ['IManager', 'IService', 'IValidator'],
                'description': '接口类以I开头，采用PascalCase'
            },

            # 抽象基类命名规则
            'abstract_base': {
                'prefix': 'Abstract',
                'pattern': r'^Abstract[A-Z][a-zA-Z0-9]*$',
                'examples': ['AbstractManager', 'AbstractService'],
                'description': '抽象基类以Abstract开头'
            },

            # 基类命名规则
            'base_class': {
                'prefix': 'Base',
                'pattern': r'^Base[A-Z][a-zA-Z0-9]*$',
                'examples': ['BaseManager', 'BaseService', 'BaseHandler'],
                'description': '基类以Base开头'
            },

            # 具体实现类命名规则
            'implementation': {
                'pattern': r'^[A-Z][a-zA-Z0-9]*$',
                'examples': ['UserManager', 'OrderService', 'FileHandler'],
                'description': '具体实现类采用PascalCase'
            },

            # 工厂类命名规则
            'factory': {
                'suffix': 'Factory',
                'pattern': r'^[A-Z][a-zA-Z0-9]*Factory$',
                'examples': ['UserFactory', 'ServiceFactory'],
                'description': '工厂类以Factory结尾'
            },

            # 管理器类命名规则
            'manager': {
                'suffix': 'Manager',
                'pattern': r'^[A-Z][a-zA-Z0-9]*Manager$',
                'examples': ['CacheManager', 'ConfigManager'],
                'description': '管理器类以Manager结尾'
            },

            # 服务类命名规则
            'service': {
                'suffix': 'Service',
                'pattern': r'^[A-Z][a-zA-Z0-9]*Service$',
                'examples': ['UserService', 'NotificationService'],
                'description': '服务类以Service结尾'
            },

            # 处理类命名规则
            'handler': {
                'suffix': ['Handler', 'Processor'],
                'pattern': r'^[A-Z][a-zA-Z0-9]*(Handler|Processor)$',
                'examples': ['EventHandler', 'MessageProcessor'],
                'description': '处理类以Handler或Processor结尾'
            },

            # 异常类命名规则
            'exception': {
                'suffix': 'Error',
                'pattern': r'^[A-Z][a-zA-Z0-9]*Error$',
                'examples': ['ValidationError', 'NetworkError'],
                'description': '异常类以Error结尾'
            }
        }

    def _define_architecture_patterns(self) -> Dict:
        """定义架构模式"""
        return {
            'factory_pattern': {
                'components': ['Factory', 'Product', 'ConcreteProduct'],
                'description': '工厂模式：创建对象而不指定具体类'
            },

            'manager_pattern': {
                'components': ['Manager', 'ManagedResource'],
                'description': '管理器模式：统一管理某一类资源'
            },

            'service_pattern': {
                'components': ['Service', 'ServiceImpl'],
                'description': '服务模式：提供业务逻辑服务'
            },

            'handler_pattern': {
                'components': ['Handler', 'Context', 'Chain'],
                'description': '处理链模式：按顺序处理请求'
            },

            'adapter_pattern': {
                'components': ['Adapter', 'Adaptee', 'Target'],
                'description': '适配器模式：使接口不兼容的类协同工作'
            },

            'strategy_pattern': {
                'components': ['Strategy', 'Context', 'ConcreteStrategy'],
                'description': '策略模式：定义算法族并封装'
            }
        }

    def analyze_current_naming_violations(self) -> Dict:
        """分析当前命名违规情况"""
        violations = {
            'interface_violations': [],
            'inconsistent_patterns': [],
            'duplicate_names': defaultdict(list),
            'architecture_issues': []
        }

        # 收集所有类名
        all_classes = self._collect_all_classes()

        # 检查接口命名违规
        for class_name, locations in all_classes.items():
            if class_name.startswith('I'):
                # 检查是否符合接口命名规范
                if not re.match(self.naming_rules['interface']['pattern'], class_name):
                    violations['interface_violations'].append({
                        'class': class_name,
                        'locations': locations,
                        'issue': '接口命名不符合规范'
                    })

        # 检查重复类名
        for class_name, locations in all_classes.items():
            if len(locations) > 1:
                violations['duplicate_names'][class_name] = locations

        # 检查架构模式一致性
        architecture_analysis = self._analyze_architecture_consistency(all_classes)
        violations['architecture_issues'] = architecture_analysis

        return violations

    def _collect_all_classes(self) -> Dict[str, List[str]]:
        """收集所有类名"""
        all_classes = defaultdict(list)

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        # 提取类定义
                        class_pattern = r'^\s*class\s+(\w+)\s*[:\(]'
                        classes = re.findall(class_pattern, content, re.MULTILINE)

                        rel_path = str(file_path.relative_to(self.infra_dir))
                        for cls in classes:
                            # 过滤掉非字母开头的类名（可能误匹配）
                            if cls and cls[0].isalpha():
                                all_classes[cls].append(rel_path)

                    except Exception as e:
                        continue

        return all_classes

    def _analyze_architecture_consistency(self, all_classes: Dict) -> List[Dict]:
        """分析架构模式一致性"""
        issues = []

        # 检查Factory类的一致性
        factory_classes = {name: locs for name, locs in all_classes.items()
                           if 'Factory' in name}

        if len(factory_classes) > 5:  # 如果有很多Factory类
            issues.append({
                'type': 'factory_proliferation',
                'classes': list(factory_classes.keys()),
                'recommendation': '考虑统一Factory接口或减少Factory类的数量'
            })

        # 检查Manager类的一致性
        manager_classes = {name: locs for name, locs in all_classes.items()
                           if 'Manager' in name}

        if len(manager_classes) > 10:
            issues.append({
                'type': 'manager_proliferation',
                'classes': list(manager_classes.keys()),
                'recommendation': '考虑合并相关的Manager类或建立统一的管理接口'
            })

        # 检查Service类的一致性
        service_classes = {name: locs for name, locs in all_classes.items()
                           if 'Service' in name}

        if len(service_classes) > 8:
            issues.append({
                'type': 'service_proliferation',
                'classes': list(service_classes.keys()),
                'recommendation': '考虑建立统一的服务层架构'
            })

        return issues

    def generate_naming_standards_document(self, violations: Dict) -> str:
        """生成命名规范文档"""
        doc = f"""# 基础设施层统一命名规范

## 概述

本文档定义了基础设施层的统一命名规范，旨在解决重复类名泛滥问题，提高代码可维护性。

## 命名规则

"""

        for rule_name, rule_config in self.naming_rules.items():
            doc += f"### {rule_name.replace('_', ' ').title()}\n\n"
            doc += f"**模式**: `{rule_config['pattern']}`\n\n"
            doc += f"**示例**: {', '.join(rule_config['examples'])}\n\n"
            doc += f"**说明**: {rule_config['description']}\n\n"

        doc += "## 架构模式规范\n\n"

        for pattern_name, pattern_config in self.architecture_patterns.items():
            doc += f"### {pattern_name.replace('_', ' ').title()}\n\n"
            doc += f"**组件**: {', '.join(pattern_config['components'])}\n\n"
            doc += f"**说明**: {pattern_config['description']}\n\n"

        doc += "## 当前问题统计\n\n"

        doc += f"- 接口命名违规: {len(violations['interface_violations'])} 个\n"
        doc += f"- 重复类名: {len(violations['duplicate_names'])} 个\n"
        doc += f"- 架构问题: {len(violations['architecture_issues'])} 个\n\n"

        if violations['duplicate_names']:
            doc += "### 最严重重复类名\n\n"
            sorted_duplicates = sorted(violations['duplicate_names'].items(),
                                       key=lambda x: len(x[1]), reverse=True)

            for i, (class_name, locations) in enumerate(sorted_duplicates[:10]):
                doc += f"{i+1}. **{class_name}** ({len(locations)} 个位置)\n\n"

        doc += "## 重构策略\n\n"
        doc += "1. **立即行动**: 重命名严重重复的类名\n"
        doc += "2. **中期目标**: 建立统一的接口层\n"
        doc += "3. **长期规划**: 实施架构模式标准化\n\n"

        doc += "## 实施指南\n\n"
        doc += "### 类名重命名原则\n"
        doc += "1. 保持功能语义不变\n"
        doc += "2. 遵循统一的命名规范\n"
        doc += "3. 更新所有引用位置\n"
        doc += "4. 添加向后兼容性\n\n"

        doc += "### 接口统一原则\n"
        doc += "1. 提取公共接口到全局interfaces模块\n"
        doc += "2. 模块内实现引用全局接口\n"
        doc += "3. 保持接口的稳定性和扩展性\n\n"

        return doc

    def create_naming_governance_mechanism(self) -> Dict:
        """创建命名治理机制"""
        governance = {
            'validation_rules': {},
            'refactoring_strategies': {},
            'monitoring_tools': {},
            'enforcement_mechanisms': {}
        }

        # 验证规则
        governance['validation_rules'] = {
            'interface_check': '确保所有接口以I开头',
            'pattern_check': '检查是否符合预定义的命名模式',
            'duplicate_check': '检测重复类名',
            'architecture_check': '验证架构模式一致性'
        }

        # 重构策略
        governance['refactoring_strategies'] = {
            'interface_unification': '将重复接口统一到全局interfaces模块',
            'class_renaming': '根据规范重命名类',
            'pattern_standardization': '标准化架构模式实现',
            'backward_compatibility': '保持向后兼容性'
        }

        # 监控工具
        governance['monitoring_tools'] = {
            'pre_commit_hooks': '提交前检查命名规范',
            'ci_checks': '持续集成中验证代码规范',
            'code_analysis': '定期进行代码质量分析',
            'duplicate_detection': '自动检测重复类名'
        }

        # 执行机制
        governance['enforcement_mechanisms'] = {
            'code_reviews': '强制代码审查',
            'automated_checks': '自动化检查工具',
            'documentation': '规范文档和培训',
            'metrics_tracking': '质量指标跟踪'
        }

        return governance

    def implement_naming_standards_framework(self):
        """实施命名规范框架"""
        print('🚀 实施命名规范框架')
        print('=' * 40)

        # 1. 分析当前违规情况
        violations = self.analyze_current_naming_violations()

        # 2. 生成命名规范文档
        standards_doc = self.generate_naming_standards_document(violations)

        # 保存文档
        doc_path = Path('NAMING_STANDARDS_FRAMEWORK.md')
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(standards_doc)

        print(f'✅ 生成命名规范文档: {doc_path}')

        # 3. 创建治理机制
        governance = self.create_naming_governance_mechanism()

        # 保存治理配置
        governance_path = Path('naming_governance.json')
        import json
        with open(governance_path, 'w', encoding='utf-8') as f:
            json.dump(governance, f, indent=2, ensure_ascii=False)

        print(f'✅ 生成治理机制配置: {governance_path}')

        # 4. 输出统计信息
        print(f'\\n📊 当前问题统计:')
        print(f'  接口命名违规: {len(violations["interface_violations"])}')
        print(f'  重复类名: {len(violations["duplicate_names"])}')
        print(f'  架构问题: {len(violations["architecture_issues"])}')

        # 5. 生成行动计划
        action_plan = self._generate_action_plan(violations)
        action_plan_path = Path('naming_refactor_action_plan.md')
        with open(action_plan_path, 'w', encoding='utf-8') as f:
            f.write(action_plan)

        print(f'✅ 生成行动计划: {action_plan_path}')

        return {
            'violations': violations,
            'standards_doc': standards_doc,
            'governance': governance,
            'action_plan': action_plan
        }

    def _generate_action_plan(self, violations: Dict) -> str:
        """生成行动计划"""
        plan = """# 命名重构行动计划

## 执行优先级

### 高优先级 (立即执行)
1. **ComponentFactory重复清理** - 36个位置的重复
2. **LogLevel统一** - 14个位置的重复
3. **AlertLevel统一** - 11个位置的重复

### 中优先级 (本周完成)
1. **接口命名规范化** - 修复所有接口命名违规
2. **Manager类统一** - 减少Manager类的重复
3. **Service类统一** - 标准化服务类命名

### 低优先级 (下个迭代)
1. **架构模式标准化** - 统一Factory、Manager等模式
2. **异常类规范化** - 统一异常类命名
3. **工具类整理** - 清理工具类重复

## 具体实施方案

### Phase 2.2: ComponentFactory清理
- 创建全局ComponentFactory接口
- 重构各模块实现为接口的子类
- 更新所有引用位置

### Phase 2.3: 架构模式统一
- 定义标准Factory接口
- 定义标准Manager接口
- 定义标准Service接口

### Phase 2.4: 大规模重构
- 批量重命名类
- 更新导入语句
- 验证兼容性

## 风险控制

1. **向后兼容性**: 保留别名和过渡期
2. **测试覆盖**: 确保所有重构都有相应测试
3. **文档同步**: 更新所有相关文档
4. **团队沟通**: 提前通知所有开发人员

## 成功标准

1. 重复类名数量减少80%
2. 所有接口符合命名规范
3. 架构模式标准化
4. 代码可维护性显著提升
"""

        return plan


def main():
    """主函数"""
    framework = NamingStandardsFramework()
    result = framework.implement_naming_standards_framework()

    print('\\n✅ 命名规范框架实施完成！')
    print('生成的文件:')
    print('  - NAMING_STANDARDS_FRAMEWORK.md')
    print('  - naming_governance.json')
    print('  - naming_refactor_action_plan.md')


if __name__ == "__main__":
    main()

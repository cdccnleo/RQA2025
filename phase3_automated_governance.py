"""
Phase 3.2: 自动化治理落实工具

实施代码审查自动化，建立CI/CD质量门禁
"""

import os
import re
from pathlib import Path
from typing import Dict, List
import json


class AutomatedCodeReview:
    """自动化代码审查工具"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.issues = []
        self.passed_checks = 0
        self.failed_checks = 0

    def run_full_review(self) -> Dict:
        """运行完整代码审查"""
        print('🔍 开始自动化代码审查...')
        print('=' * 50)

        # 1. 导入规范检查
        self.check_import_standards()

        # 2. 命名规范检查
        self.check_naming_standards()

        # 3. 架构模式检查
        self.check_architecture_patterns()

        # 4. 代码质量检查
        self.check_code_quality()

        # 5. 接口实现检查
        self.check_interface_implementation()

        # 6. 依赖关系检查
        self.check_dependencies()

        # 生成报告
        report = self.generate_review_report()

        print(f'\\n📊 审查结果:')
        print(f'  通过检查: {self.passed_checks}')
        print(f'  失败检查: {self.failed_checks}')
        print(f'  发现问题: {len(self.issues)}')

        return report

    def check_import_standards(self):
        """检查导入规范"""
        print('\\n📦 检查导入规范...')

        wildcard_imports = []
        long_imports = []
        unordered_imports = []

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查通配符导入
                        for line_num, line in enumerate(lines[:30]):
                            line = line.strip()
                            if 'from ' in line and ' import *' in line:
                                wildcard_imports.append(f'{rel_path}:{line_num+1}')

                        # 检查过长导入
                        for line_num, line in enumerate(lines[:30]):
                            line = line.strip()
                            if line.startswith('from ') and len(line) > 100:
                                long_imports.append(f'{rel_path}:{line_num+1}')

                        # 检查导入顺序（简化的检查）
                        import_lines = []
                        for line in lines[:30]:
                            line = line.strip()
                            if line.startswith('import ') or line.startswith('from '):
                                import_lines.append(line)

                        # 检查是否有明显的顺序问题
                        if len(import_lines) > 3:
                            stdlib_imports = [l for l in import_lines if not l.startswith(
                                'from src.') and not '.' in l.split()[1].split('.')[0]]
                            local_imports = [l for l in import_lines if l.startswith('from src.')]

                            if local_imports and stdlib_imports and import_lines.index(local_imports[0]) < import_lines.index(stdlib_imports[-1]):
                                unordered_imports.append(rel_path)

                    except Exception as e:
                        continue

        # 记录问题
        if wildcard_imports:
            self.issues.append({
                'type': 'import_violation',
                'category': 'wildcard_imports',
                'severity': 'high',
                'count': len(wildcard_imports),
                'details': wildcard_imports[:10]
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

        if long_imports:
            self.issues.append({
                'type': 'import_violation',
                'category': 'long_imports',
                'severity': 'medium',
                'count': len(long_imports),
                'details': long_imports[:10]
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

        if unordered_imports:
            self.issues.append({
                'type': 'import_violation',
                'category': 'unordered_imports',
                'severity': 'low',
                'count': len(unordered_imports),
                'details': unordered_imports[:10]
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

    def check_naming_standards(self):
        """检查命名规范"""
        print('\\n🏷️ 检查命名规范...')

        naming_violations = {
            'interface_naming': [],
            'class_naming': [],
            'method_naming': [],
            'variable_naming': []
        }

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查接口命名
                        interface_matches = re.findall(r'class\s+(I\w+)', content)
                        for interface in interface_matches:
                            if not re.match(r'^I[A-Z][a-zA-Z0-9]*$', interface):
                                naming_violations['interface_naming'].append(
                                    f'{rel_path}: {interface}')

                        # 检查类命名
                        class_matches = re.findall(r'class\s+([A-Z]\w+)', content)
                        for cls in class_matches:
                            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', cls):
                                naming_violations['class_naming'].append(f'{rel_path}: {cls}')

                    except Exception as e:
                        continue

        # 记录问题
        for category, violations in naming_violations.items():
            if violations:
                self.issues.append({
                    'type': 'naming_violation',
                    'category': category,
                    'severity': 'medium',
                    'count': len(violations),
                    'details': violations[:10]
                })
                self.failed_checks += 1
            else:
                self.passed_checks += 1

    def check_architecture_patterns(self):
        """检查架构模式"""
        print('\\n🏗️ 检查架构模式...')

        pattern_violations = []

        # 检查是否有未使用统一接口的类
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查Factory类是否继承了统一接口
                        if 'class ' in content and 'Factory' in content:
                            class_lines = [line for line in content.split(
                                '\n') if 'class ' in line and 'Factory' in line]
                            for line in class_lines:
                                if 'BaseComponentFactory' not in line and 'BaseFactory' not in line:
                                    pattern_violations.append(f'{rel_path}: {line.strip()}')

                    except Exception as e:
                        continue

        if pattern_violations:
            self.issues.append({
                'type': 'architecture_violation',
                'category': 'missing_interface_inheritance',
                'severity': 'high',
                'count': len(pattern_violations),
                'details': pattern_violations[:10]
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

    def check_code_quality(self):
        """检查代码质量"""
        print('\\n🔍 检查代码质量...')

        quality_issues = {
            'long_functions': [],
            'complex_functions': [],
            'missing_docstrings': []
        }

        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查长函数（简化检查）
                        functions = re.findall(r'def\s+\w+.*?:', content)
                        if len(functions) > 20:  # 文件中有太多函数
                            quality_issues['long_functions'].append(
                                f'{rel_path}: {len(functions)} functions')

                        # 检查缺少文档字符串的函数（简化检查）
                        func_blocks = re.findall(
                            r'def\s+\w+.*?:.*?(?=\\n\\n|\\n\s*def|\\n\s*@|\\n\s*class|\\Z)', content, re.DOTALL)
                        for block in func_blocks[:5]:  # 只检查前5个函数
                            if '"""' not in block and "'''" not in block:
                                func_name = re.search(r'def\s+(\w+)', block)
                                if func_name:
                                    quality_issues['missing_docstrings'].append(
                                        f'{rel_path}: {func_name.group(1)}')

                    except Exception as e:
                        continue

        # 记录问题
        for category, issues in quality_issues.items():
            if issues:
                severity = 'low' if category == 'missing_docstrings' else 'medium'
                self.issues.append({
                    'type': 'quality_violation',
                    'category': category,
                    'severity': severity,
                    'count': len(issues),
                    'details': issues[:10]
                })
                self.failed_checks += 1
            else:
                self.passed_checks += 1

    def check_interface_implementation(self):
        """检查接口实现"""
        print('\\n🔗 检查接口实现...')

        interface_issues = []

        # 检查是否有实现了接口但缺少必要方法的类
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查是否实现了BaseComponentFactory但缺少_create_component_instance方法
                        if 'BaseComponentFactory' in content:
                            if '_create_component_instance' not in content:
                                interface_issues.append(
                                    f'{rel_path}: BaseComponentFactory缺少_create_component_instance方法')

                    except Exception as e:
                        continue

        if interface_issues:
            self.issues.append({
                'type': 'interface_violation',
                'category': 'missing_interface_methods',
                'severity': 'high',
                'count': len(interface_issues),
                'details': interface_issues
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

    def check_dependencies(self):
        """检查依赖关系"""
        print('\\n🔗 检查依赖关系...')

        dependency_issues = []

        # 检查循环依赖（简化检查）
        for root, dirs, files in os.walk(self.infra_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        rel_path = str(file_path.relative_to(self.infra_dir))

                        # 检查可能的循环依赖模式
                        imports = re.findall(r'from\s+src\.infrastructure\.(\w+)', content)
                        self_imports = [imp for imp in imports if imp in rel_path.split('/')[0]]

                        if len(self_imports) > 3:  # 同一个模块内过多相互导入
                            dependency_issues.append(
                                f'{rel_path}: 可能的循环依赖 ({len(self_imports)} 个内部导入)')

                    except Exception as e:
                        continue

        if dependency_issues:
            self.issues.append({
                'type': 'dependency_violation',
                'category': 'potential_circular_imports',
                'severity': 'medium',
                'count': len(dependency_issues),
                'details': dependency_issues
            })
            self.failed_checks += 1
        else:
            self.passed_checks += 1

    def generate_review_report(self) -> Dict:
        """生成审查报告"""
        report = {
            'summary': {
                'total_checks': self.passed_checks + self.failed_checks,
                'passed_checks': self.passed_checks,
                'failed_checks': self.failed_checks,
                'total_issues': len(self.issues),
                'pass_rate': self.passed_checks / (self.passed_checks + self.failed_checks) * 100 if (self.passed_checks + self.failed_checks) > 0 else 0
            },
            'issues': self.issues,
            'recommendations': self.generate_recommendations()
        }

        return report

    def generate_recommendations(self) -> List[str]:
        """生成修复建议"""
        recommendations = []

        # 根据发现的问题生成建议
        for issue in self.issues:
            if issue['category'] == 'wildcard_imports':
                recommendations.append(f"移除 {issue['count']} 个通配符导入，使用显式导入")
            elif issue['category'] == 'long_imports':
                recommendations.append(f"拆分 {issue['count']} 个过长导入语句")
            elif issue['category'] == 'missing_interface_inheritance':
                recommendations.append(f"修复 {issue['count']} 个未继承统一接口的类")
            elif issue['category'] == 'missing_interface_methods':
                recommendations.append(f"实现 {issue['count']} 个缺失的接口方法")

        # 通用建议
        recommendations.extend([
            "建立pre-commit钩子自动检查代码规范",
            "集成CI/CD流水线进行自动化代码审查",
            "定期进行代码质量分析和改进",
            "建立团队代码规范培训机制"
        ])

        return recommendations

    def save_report(self, report: Dict, output_file: str = 'automated_review_report.json'):
        """保存报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f'\\n✅ 审查报告已保存到: {output_file}')


class GovernanceAutomation:
    """治理自动化"""

    def __init__(self):
        self.review_tool = AutomatedCodeReview()

    def run_automated_governance(self):
        """运行自动化治理"""
        print('🚀 开始Phase 3.2: 自动化治理落实')
        print('=' * 50)

        # 1. 运行自动化审查
        report = self.review_tool.run_full_review()

        # 2. 保存报告
        self.review_tool.save_report(report)

        # 3. 生成治理配置
        governance_config = self.create_governance_config(report)

        # 4. 创建CI/CD配置
        ci_config = self.create_ci_config(report)

        return {
            'review_report': report,
            'governance_config': governance_config,
            'ci_config': ci_config
        }

    def create_governance_config(self, report: Dict) -> Dict:
        """创建治理配置"""
        config = {
            'version': '1.0',
            'enabled_checks': [
                'import_standards',
                'naming_conventions',
                'architecture_patterns',
                'code_quality',
                'interface_implementation',
                'dependency_analysis'
            ],
            'severity_levels': {
                'high': ['wildcard_imports', 'missing_interface_inheritance', 'missing_interface_methods'],
                'medium': ['long_imports', 'naming_violations', 'architecture_violations'],
                'low': ['unordered_imports', 'quality_violations']
            },
            'automated_fixes': {
                'import_sorting': True,
                'naming_standardization': False,  # 需要人工确认
                'interface_inheritance': False   # 需要人工实现
            },
            'reporting': {
                'console_output': True,
                'json_report': True,
                'html_report': False,
                'email_notifications': False
            }
        }

        # 保存配置
        with open('governance_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print('✅ 治理配置已保存: governance_config.json')
        return config

    def create_ci_config(self, report: Dict) -> str:
        """创建CI/CD配置"""
        ci_config = '''# CI/CD 代码质量检查配置
# 此文件应放置在项目根目录，作为GitHub Actions或其他CI工具的配置

name: Code Quality Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Run automated code review
      run: |
        python phase3_automated_governance.py

    - name: Check code quality
      run: |
        # 这里可以添加更多的质量检查
        python -m py_compile src/infrastructure/**/*.py

    - name: Upload review results
      uses: actions/upload-artifact@v3
      with:
        name: code-review-results
        path: |
          automated_review_report.json
          governance_config.json
'''

        with open('.github/workflows/code-quality.yml', 'w', encoding='utf-8') as f:
            f.write(ci_config)

        print('✅ CI/CD配置已创建: .github/workflows/code-quality.yml')
        return ci_config

    def create_pre_commit_hook(self) -> str:
        """创建pre-commit钩子"""
        hook_script = '''#!/bin/bash
# Pre-commit hook for code quality checks

echo "Running pre-commit code quality checks..."

# Run automated review
python phase3_automated_governance.py

# Check if there are any critical issues
if [ -f "automated_review_report.json" ]; then
    # Parse the report and check for high-severity issues
    CRITICAL_ISSUES=$(python -c "
import json
with open('automated_review_report.json') as f:
    report = json.load(f)
critical_count = sum(1 for issue in report['issues'] if issue['severity'] == 'high')
print(critical_count)
")

    if [ "$CRITICAL_ISSUES" -gt 0 ]; then
        echo "❌ Found $CRITICAL_ISSUES critical code quality issues!"
        echo "Please fix them before committing."
        exit 1
    fi
fi

echo "✅ All pre-commit checks passed!"
exit 0
'''

        # 创建pre-commit钩子目录
        hook_dir = Path('.git/hooks')
        hook_dir.mkdir(exist_ok=True)

        hook_path = hook_dir / 'pre-commit'
        with open(hook_path, 'w', encoding='utf-8') as f:
            f.write(hook_script)

        # 设置执行权限
        import stat
        hook_path.chmod(hook_path.stat().st_mode | stat.S_IEXEC)

        print('✅ Pre-commit钩子已创建: .git/hooks/pre-commit')
        return hook_script


def main():
    """主函数"""
    governance = GovernanceAutomation()

    # 运行自动化治理
    result = governance.run_automated_governance()

    # 创建pre-commit钩子
    governance.create_pre_commit_hook()

    print('\\n✅ Phase 3.2 自动化治理落实完成！')
    print('生成的文件:')
    print('  - automated_review_report.json')
    print('  - governance_config.json')
    print('  - .github/workflows/code-quality.yml')
    print('  - .git/hooks/pre-commit')


if __name__ == "__main__":
    main()

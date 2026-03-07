#!/usr/bin/env python3
"""
简单接口继承修复工具

专门修复最基本的接口继承问题，避免复杂语法错误
"""

import re
from pathlib import Path
from typing import Dict, Any


class SimpleInterfaceFixer:
    """简单接口继承修复器"""

    def __init__(self):
        self.infra_dir = Path('src/infrastructure')
        self.fixed_count = 0
        self.error_count = 0

    def fix_basic_interface_inheritance(self) -> Dict[str, Any]:
        """修复基本的接口继承问题"""
        print('🔧 开始修复基本的接口继承问题')
        print('=' * 60)

        # 手动指定的修复规则（避免读取有问题的报告文件）
        fix_rules = [
            {
                'file': 'src/infrastructure/cache/monitoring/combined_monitoring.py',
                'class': 'MockFailureCacheManager',
                'base': 'BaseManager'
            },
            {
                'file': 'src/infrastructure/cache/strategies/unified_strategy_manager.py',
                'class': 'SmartCacheManager',
                'base': 'BaseManager'
            },
            {
                'file': 'src/infrastructure/config/interfaces.py',
                'class': 'IConfigManager',
                'base': 'BaseManager'
            },
            {
                'file': 'src/infrastructure/config/core/core_components.py',
                'class': 'ServiceStatus',
                'base': 'BaseService'
            },
            # 添加更多规则...
        ]

        results = {
            'total_fixes_attempted': len(fix_rules),
            'successful_fixes': 0,
            'failed_fixes': 0,
            'details': []
        }

        for rule in fix_rules:
            try:
                result = self._fix_single_rule(rule)
                if result['success']:
                    results['successful_fixes'] += 1
                    results['details'].append(result)
                    print(f"✅ 修复: {rule['class']} -> {rule['base']}")
                else:
                    results['failed_fixes'] += 1
                    print(f"❌ 修复失败: {rule['class']} - {result.get('error', '未知错误')}")
            except Exception as e:
                results['failed_fixes'] += 1
                print(f"❌ 异常: {rule['class']} - {e}")

        print('\\n✅ 基本接口继承修复完成')
        self._print_simple_summary(results)

        return results

    def _fix_single_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        """修复单个规则"""
        result = {
            'file': rule['file'],
            'class': rule['class'],
            'base': rule['base'],
            'success': False,
            'error': None
        }

        file_path = Path(rule['file'])
        if not file_path.exists():
            result['error'] = '文件不存在'
            return result

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否已经是接口（以I开头或Base开头）
            class_name = rule['class']
            if class_name.startswith(('I', 'Base')):
                # 接口类不需要继承，只需要确保导入正确
                result['success'] = True
                result['note'] = '接口类无需继承'
                return result

            # 检查类定义是否存在
            class_pattern = rf'class\s+{re.escape(class_name)}\s*\('
            if not re.search(class_pattern, content):
                result['error'] = '类定义不存在'
                return result

            # 检查是否已经继承了该基类
            inheritance_pattern = rf'class\s+{re.escape(class_name)}\s*\([^)]*{re.escape(rule["base"])}[^)]*\)'
            if re.search(inheritance_pattern, content):
                result['success'] = True
                result['note'] = '已经正确继承'
                return result

            # 修复继承关系
            success = self._add_base_inheritance(content, file_path, class_name, rule['base'])
            if success:
                result['success'] = True
            else:
                result['error'] = '修复继承失败'

        except Exception as e:
            result['error'] = str(e)

        return result

    def _add_base_inheritance(self, content: str, file_path: Path, class_name: str, base_class: str) -> bool:
        """添加基类继承"""
        try:
            # 查找类定义
            class_pattern = rf'(class\s+{re.escape(class_name)}\s*\()([^)]*)(\))'
            match = re.search(class_pattern, content, re.MULTILINE)

            if not match:
                return False

            full_match = match.group(0)
            before_paren = match.group(1)
            inheritance_list = match.group(2).strip()

            # 构建新的继承列表
            if inheritance_list:
                new_inheritance = f'{inheritance_list}, {base_class}'
            else:
                new_inheritance = base_class

            new_class_def = f'{before_paren}{new_inheritance})'

            # 替换类定义
            content = content.replace(full_match, new_class_def)

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # 验证语法
            import py_compile
            py_compile.compile(str(file_path), doraise=True)

            return True

        except Exception:
            return False

    def _print_simple_summary(self, results: Dict[str, Any]):
        """打印简单总结"""
        print('\\n🔧 简单接口继承修复总结:')
        print('-' * 50)
        print(f'📋 修复尝试数: {results["total_fixes_attempted"]}')
        print(f'✅ 修复成功: {results["successful_fixes"]}')
        print(f'❌ 修复失败: {results["failed_fixes"]}')

        success_rate = (results['successful_fixes'] / results['total_fixes_attempted']
                        * 100) if results['total_fixes_attempted'] > 0 else 0
        print('.1f')

        if results['details']:
            print('\\n✅ 成功修复的类:')
            for detail in results['details'][:5]:  # 显示前5个
                print(f'   • {detail["class"]} -> {detail["base"]}')

        print('\\n📄 修复报告已保存: simple_interface_fix_report.json')


def main():
    """主函数"""
    fixer = SimpleInterfaceFixer()
    results = fixer.fix_basic_interface_inheritance()

    # 保存报告
    with open('simple_interface_fix_report.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

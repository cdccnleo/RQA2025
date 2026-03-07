#!/usr/bin/env python3
"""
RQA2025 错误分析脚本

分析测试错误并提供修复建议
"""

import subprocess
import re
from pathlib import Path
from datetime import datetime


class ErrorAnalyzer:
    """错误分析器"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent

    def analyze_layer_errors(self, layer_name: str):
        """分析指定层的错误"""
        print(f"\n🔍 分析 {layer_name.upper()} 层错误...")

        test_path = f"tests/unit/{layer_name}/"

        # 运行测试并捕获详细错误
        cmd = [
            'conda', 'run', '-n', 'rqa', 'python', '-m', 'pytest',
            test_path,
            '-v',
            '--tb=short'
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )

            output = result.stdout + result.stderr

            # 分析错误
            errors = self.extract_errors(output)

            print(f"📊 {layer_name.upper()} 层错误分析:")
            print(f"   - 总错误数: {len(errors)}")

            # 分类错误
            error_types = self.categorize_errors(errors)

            for error_type, count in error_types.items():
                print(f"   - {error_type}: {count}")

            # 显示前5个错误
            print(f"\n📋 前5个错误详情:")
            for i, error in enumerate(errors[:5]):
                print(f"   {i+1}. {error}")

            return errors

        except Exception as e:
            print(f"❌ 分析 {layer_name} 层时出错: {e}")
            return []

    def extract_errors(self, output: str) -> list:
        """提取错误信息"""
        errors = []

        # 查找ImportError
        import_errors = re.findall(r'ImportError.*?(?=\n\n|\n[A-Z]|\n$)', output, re.DOTALL)
        errors.extend(import_errors)

        # 查找ModuleNotFoundError
        module_errors = re.findall(r'ModuleNotFoundError.*?(?=\n\n|\n[A-Z]|\n$)', output, re.DOTALL)
        errors.extend(module_errors)

        # 查找SyntaxError
        syntax_errors = re.findall(r'SyntaxError.*?(?=\n\n|\n[A-Z]|\n$)', output, re.DOTALL)
        errors.extend(syntax_errors)

        # 查找AttributeError
        attr_errors = re.findall(r'AttributeError.*?(?=\n\n|\n[A-Z]|\n$)', output, re.DOTALL)
        errors.extend(attr_errors)

        # 查找其他错误
        other_errors = re.findall(r'ERROR.*?(?=\n\n|\n[A-Z]|\n$)', output, re.DOTALL)
        errors.extend(other_errors)

        return errors

    def categorize_errors(self, errors: list) -> dict:
        """分类错误"""
        categories = {
            'ImportError': 0,
            'ModuleNotFoundError': 0,
            'SyntaxError': 0,
            'AttributeError': 0,
            '其他': 0
        }

        for error in errors:
            if 'ImportError' in error:
                categories['ImportError'] += 1
            elif 'ModuleNotFoundError' in error:
                categories['ModuleNotFoundError'] += 1
            elif 'SyntaxError' in error:
                categories['SyntaxError'] += 1
            elif 'AttributeError' in error:
                categories['AttributeError'] += 1
            else:
                categories['其他'] += 1

        return categories

    def generate_fix_suggestions(self, errors: list) -> list:
        """生成修复建议"""
        suggestions = []

        for error in errors:
            if 'ModuleNotFoundError' in error:
                # 提取模块名
                match = re.search(r"No module named '([^']+)'", error)
                if match:
                    module_name = match.group(1)
                    suggestions.append(f"安装缺失模块: pip install {module_name}")

            elif 'ImportError' in error:
                # 提取导入路径
                match = re.search(r"cannot import name '([^']+)'", error)
                if match:
                    import_name = match.group(1)
                    suggestions.append(f"修复导入: {import_name}")

            elif 'SyntaxError' in error:
                suggestions.append("检查语法错误，可能需要修复代码格式")

        return suggestions

    def analyze_all_layers(self):
        """分析所有层"""
        layers = ['infrastructure', 'data', 'features', 'models', 'trading', 'backtest']

        all_errors = {}
        all_suggestions = []

        for layer in layers:
            errors = self.analyze_layer_errors(layer)
            all_errors[layer] = errors

            suggestions = self.generate_fix_suggestions(errors)
            all_suggestions.extend(suggestions)

        # 生成总结报告
        self.generate_summary_report(all_errors, all_suggestions)

    def generate_summary_report(self, all_errors: dict, all_suggestions: list):
        """生成总结报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / f"reports/error_analysis_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 错误分析报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 错误统计\n\n")
            total_errors = sum(len(errors) for errors in all_errors.values())
            f.write(f"- **总错误数**: {total_errors}\n\n")

            f.write("## 各层错误详情\n\n")
            for layer, errors in all_errors.items():
                f.write(f"### {layer.upper()} 层\n")
                f.write(f"- **错误数**: {len(errors)}\n")
                if errors:
                    f.write("- **主要错误**:\n")
                    for i, error in enumerate(errors[:3]):
                        f.write(f"  {i+1}. {error[:100]}...\n")
                f.write("\n")

            f.write("## 修复建议\n\n")
            unique_suggestions = list(set(all_suggestions))
            for i, suggestion in enumerate(unique_suggestions):
                f.write(f"{i+1}. {suggestion}\n")

        print(f"📄 错误分析报告已保存到: {report_file}")


def main():
    """主函数"""
    analyzer = ErrorAnalyzer()
    analyzer.analyze_all_layers()


if __name__ == "__main__":
    main()
